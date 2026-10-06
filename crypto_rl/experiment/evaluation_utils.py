# -*- coding: utf-8 -*-
"""Evaluation utilities for experiment: baseline calculations and multi-seed evaluation."""

import dataclasses
import logging
from typing import Any

import numpy as np
import optuna
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.config import RLConfig
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.logging_utils import LoggerBase
from crypto_rl.env.minimal_env import MinimalCryptoEnv


def compute_buy_and_hold_baseline(
    config: RLConfig,
    last_asset_names: list[str],
    last_prices_arr: np.ndarray,
) -> float:
    """Calculate the buy-and-hold final portfolio value benchmark.

    Selects BTCUSDT if present, otherwise locates the first symbol with start_price > 0.
    """
    if "BTCUSDT" in last_asset_names:
        btc_idx = last_asset_names.index("BTCUSDT")
        start_price = last_prices_arr[0, btc_idx]
        end_price = last_prices_arr[-1, btc_idx]
    else:
        start_price = last_prices_arr[0, 0]
        end_price = last_prices_arr[-1, 0]

    if start_price > 1e-8:
        buy_hold_return = (end_price - start_price) / start_price
    else:
        buy_hold_return = 0.0

    return float(config.budget_initial * (1 + buy_hold_return))


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
        ms_env_masked = ActionMasker(raw_ms_env, get_action_mask)
        ms_env = VecNormalize(
            DummyVecEnv([lambda: ms_env_masked]), **dummy_vec_env_args
        )
        ms_env.obs_rms = train_env_obs_rms
        ms_obs = ms_env.reset()
        ms_done = False
        final_ms_pv = None
        while not ms_done:
            action_masks = np.expand_dims(ms_env.venv.envs[0].action_masks(), axis=0)
            ms_action, _ = last_model.predict(
                ms_obs, action_masks=action_masks, deterministic=True
            )
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
