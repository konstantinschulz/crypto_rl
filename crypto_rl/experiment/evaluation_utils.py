# -*- coding: utf-8 -*-
"""Evaluation utilities for experiment: baseline calculations and multi-seed evaluation."""

import dataclasses
import logging
from typing import Any

import numpy as np
import optuna
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.config import RLConfig
from crypto_rl.env.logging_utils import LoggerBase
from crypto_rl.env.minimal_env import MinimalCryptoEnv


BASELINE_NAMES = ("equal_weight", "btc", "cash")


def _buy_and_hold_value(
    budgets: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    fee_rate: float,
) -> float:
    """Final value of spending `budgets` per asset at entry and selling at exit.

    Uses the env's fee model: a buy of S cash gets S / ((1 + fee) * price)
    units, a sell of u units returns u * price * (1 - fee).
    """
    valid = entry_prices > 1e-8
    units = np.divide(
        budgets,
        (1.0 + fee_rate) * entry_prices,
        out=np.zeros_like(budgets, dtype=float),
        where=valid,
    )
    proceeds = float(np.sum(units * exit_prices * (1.0 - fee_rate)))
    # Budget that could not be invested (no valid entry price) stays in cash
    return proceeds + float(np.sum(budgets[~valid]))


def compute_window_baselines(
    prices_arr: np.ndarray,
    start_step: int,
    max_steps: int,
    asset_names: list[str],
    fee_rate: float,
    budget_initial: float,
) -> dict[str, float]:
    """Final portfolio values of the passive baselines over the bot's window.

    An episode that starts at ``current_step == start_step`` executes its
    first action at ``prices[start_step - 1]`` and its final PV is the
    liquidation value at ``prices[max_steps - 1]``, net of exit fees (see
    `step_env`). The baselines buy at the first of those prices and sell at
    the last, with the same fees:

    - ``equal_weight``: the budget split equally across all assets.
    - ``btc``: the whole budget in BTCUSDT (first asset if it is missing).
    - ``cash``: the budget held, untouched.
    """
    entry_prices = np.asarray(prices_arr[start_step - 1], dtype=float)
    exit_prices = np.asarray(prices_arr[max_steps - 1], dtype=float)
    n_assets = len(entry_prices)

    btc_idx = asset_names.index("BTCUSDT") if "BTCUSDT" in asset_names else 0
    btc_budgets = np.zeros(n_assets)
    btc_budgets[btc_idx] = budget_initial

    return {
        "equal_weight": _buy_and_hold_value(
            np.full(n_assets, budget_initial / n_assets),
            entry_prices,
            exit_prices,
            fee_rate,
        ),
        "btc": _buy_and_hold_value(btc_budgets, entry_prices, exit_prices, fee_rate),
        "cash": float(budget_initial),
    }


def excess_returns(
    final_portfolio_value: float,
    baselines: dict[str, float],
    budget_initial: float,
) -> dict[str, float]:
    """The bot's return minus each baseline's return, as fractions of the budget."""
    return {
        name: (final_portfolio_value - value) / budget_initial
        for name, value in baselines.items()
    }


def run_multi_seed_eval(
    config: RLConfig,
    logger: LoggerBase,
    trial: optuna.trial.Trial | None,
    last_model: Any,
    last_test_prices: np.ndarray,
    last_test_static: np.ndarray,
    last_test_norm_vol: np.ndarray,
    last_test_names: list[str],
    run_id: str,
    train_env_obs_rms: Any,
    dummy_vec_env_args: dict[str, Any],
) -> list[float]:
    """Perform multi-seed evaluation on the final model across 5 seeds."""
    multi_seed_pv: list[float] = []
    if config.skip_multi_seed_eval:
        return multi_seed_pv

    logger.print_if_not_trial(
        logging.DEBUG,
        trial,
        "5. Performing multi‑seed evaluation...",
    )
    for mseed in range(5):
        np.random.seed(mseed + 100)
        raw_ms_env = MinimalCryptoEnv(
            prices_arr=last_test_prices,
            static_obs=last_test_static,
            norm_vol_arr=last_test_norm_vol,
            asset_names=last_test_names,
            run_id=run_id,
            is_eval=True,
            config=dataclasses.replace(config, disable_logging=True),
        )
        ms_env = VecNormalize(DummyVecEnv([lambda: raw_ms_env]), **dummy_vec_env_args)
        ms_env.obs_rms = train_env_obs_rms
        ms_obs = ms_env.reset()
        ms_done = False
        final_ms_pv = None
        while not ms_done:
            ms_action, _ = last_model.predict(ms_obs, deterministic=True)
            ms_obs, _, ms_dones, ms_infos = ms_env.step(ms_action)
            ms_done = ms_dones[0]
            if ms_done:
                final_ms_pv = ms_infos[0].get(
                    "final_portfolio_value", raw_ms_env.portfolio_value
                )
        if final_ms_pv is not None:
            multi_seed_pv.append(float(final_ms_pv))
        ms_env.close()

    if multi_seed_pv:
        arr = np.array(multi_seed_pv)
        logger.print_if_not_trial(
            logging.INFO,
            trial,
            f"  n={len(arr)}  mean=${arr.mean():.2f}  std=${arr.std():.2f}  "
            f"min=${arr.min():.2f}  max=${arr.max():.2f}",
        )

    return multi_seed_pv


def log_baselines(
    logger: LoggerBase,
    trial: optuna.trial.Trial | None,
    title: str,
    bot_return: float,
    baselines: dict[str, float],
    excess: dict[str, float],
    budget_initial: float,
) -> None:
    """Log each baseline's final value and return, and the bot's excess over it."""
    logger.log(title, trial)
    logger.log(f"  {'Bot':<13} | Return: {bot_return * 100:+8.2f}%", trial)
    for name in BASELINE_NAMES:
        value = baselines[name]
        ret = (value - budget_initial) / budget_initial
        logger.log(
            f"  {name:<13} | Final PV: ${value:<10.2f} | Return: {ret * 100:+8.2f}% "
            f"| Bot excess: {excess[name] * 100:+8.2f} pp",
            trial,
        )
