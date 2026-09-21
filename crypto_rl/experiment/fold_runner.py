# -*- coding: utf-8 -*-
"""Helper functions to run folds for the experiment.

This module contains the heavy‑weight loop that was previously inside
`run_experiment`.  Keeping the logic here keeps `runner.py` concise while
preserving the original behavior.
"""

import dataclasses
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.callbacks import (
    DashboardCallback,
    EntropyDecayCallback,
    UnifiedEvalCallback,
)
from crypto_rl.checkpoint_manager import CVCheckpointManager
from crypto_rl.config import RLConfig
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.data_utils import compute_static_obs_from_long_df
from crypto_rl.env.logging_utils import (
    print_if_not_trial,
    send_notification,
)
from crypto_rl.env.metrics import calculate_calmar_ratio
from crypto_rl.env.minimal_env import MinimalCryptoEnv


def run_folds(
    config: RLConfig,
    trial: optuna.trial.Trial | None,
    logger: Any,
    run_id: str,
    run_dir: Path,
    state_file: Path,
    index_file: Path,
    ckpt_mgr: CVCheckpointManager,
    shared_env_config: RLConfig,
    env_config: RLConfig,
    dummy_vec_env_args: dict[str, Any],
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[float, float, float, float, int, float, list[dict], dict, Any, list[tuple]]:
    """Execute all CV folds.

    Returns a tuple containing the aggregated metrics that the caller can use
    to update the dashboard and compute the final Calmar score.
    """
    per_asset_stats: dict[str, dict[str, float]] = {}
    fold_scores: list[float] = []
    fold_results: list[dict] = []
    last_model = None
    last_test_prices = None
    last_test_static = None
    last_test_names = None
    last_prices_arr = None
    last_asset_names: list[str] = []
    last_eval_reward_totals: dict = {}
    last_eval_portfolio_values: list = []
    last_eval_realized_pnl: list = []
    last_eval_steps = 0
    last_obs_rms = None
    evals_per_fold = config.timesteps // config.eval_freq

    # 1. Fetch completed folds if resuming a specific trial
    completed_folds: dict = {}
    if trial is not None:
        existing_ckpt = ckpt_mgr.load_checkpoint(trial.number)
        if existing_ckpt:
            completed_folds = existing_ckpt.get("completed_folds", {})

    for fold_idx, (train_prices_df, test_prices_df) in enumerate(splits):
        str_fold = str(fold_idx)
        # ----- Resume from checkpoint ------------------------------------------------
        if str_fold in completed_folds:
            cached_score = completed_folds[str_fold]["score"]
            print_if_not_trial(
                logger,
                logging.DEBUG,
                None,
                f"--> [Fold {fold_idx + 1}/{config.cv_folds}] RESUMED from checkpoint. Score: {cached_score:.4f}",
            )
            fold_scores.append(cached_score)
            if trial is not None:
                report_step = (fold_idx + 1) * evals_per_fold
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*is already reported.*",
                    )
                    trial.report(cached_score, report_step)
            continue

        # ----- Live training --------------------------------------------------------
        msg = f"--> [Fold {fold_idx + 1}/{config.cv_folds}] Executing fold..."
        print_if_not_trial(logger, logging.DEBUG, None, msg)
        send_notification(msg, summary="Experiment Run" if trial is None else "Optuna Trial")
        eval_step_offset = fold_idx * evals_per_fold

        # training period strings
        start_ts_raw = train_prices_df["open_time"].min()
        end_ts_raw = train_prices_df["open_time"].max()
        training_start_str = (
            pd.to_datetime(start_ts_raw)
            .tz_localize("UTC")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        training_end_str = (
            pd.to_datetime(end_ts_raw)
            .tz_localize("UTC")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )

        prices_arr, static_obs, norm_vol_arr, asset_names = (
            compute_static_obs_from_long_df(train_prices_df, config)
        )
        last_prices_arr = prices_arr
        last_asset_names = asset_names

        print_if_not_trial(logger, logging.DEBUG, trial, "2. Setting up environment...")

        def make_env():
            e = MinimalCryptoEnv(
                config=env_config,
                prices_arr=prices_arr,
                static_obs=static_obs,
                norm_vol_arr=norm_vol_arr,
                asset_names=asset_names,
                run_id=run_id,
            )
            return ActionMasker(e, get_action_mask)

        env_fns = [make_env for _ in range(config.n_envs)]
        train_env = VecNormalize(
            DummyVecEnv(env_fns),
            norm_reward=True,
            norm_obs=True,
            gamma=config.gamma,
            clip_obs=10.0,
            clip_reward=5.0,
        )

        # ----- Dashboard ------------------------------------------------------------
        dashboard_callback = None
        if config.dashboard and fold_idx == 0:
            try:
                index = {"runs": []}
                if index_file.exists():
                    with open(index_file, "r", encoding="utf-8") as f:
                        index = json.load(f)
                run_entry = {
                    "run_id": run_id,
                    "state_file": str(state_file),
                    "mode": "minimal",
                    "status": "initializing",
                    "started_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                runs = index.get("runs", [])
                runs.insert(0, run_entry)
                index["runs"] = runs
                index["latest_model_run"] = run_entry
                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2)
            except Exception:
                pass

        if config.dashboard:
            dashboard_callback = DashboardCallback(
                state_path=state_file,
                config=config,
                run_id=run_id,
                total_timesteps=config.timesteps,
                num_data_rows=config.n_rows,
                training_start_str=training_start_str,
                training_end_str=training_end_str,
            )

        # ----- Test observations ----------------------------------------------------
        print_if_not_trial(
            logger, logging.DEBUG, trial, "Computing test observations..."
        )
        (
            shared_test_prices,
            shared_test_static,
            shared_test_norm_vol,
            shared_test_names,
        ) = compute_static_obs_from_long_df(test_prices_df, config)

        checkpoint_dir = run_dir / f"checkpoints_fold_{fold_idx + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shared_env_args = {
            "prices_arr": shared_test_prices,
            "static_obs": shared_test_static,
            "norm_vol_arr": shared_test_norm_vol,
            "asset_names": shared_test_names,
            "run_id": run_id,
            "is_eval": True,
            "config": shared_env_config,
        }
        eval_callback = None
        eval_env = None
        if config.checkpoint or trial is not None:
            raw_eval_env = MinimalCryptoEnv(**shared_env_args)
            masked_eval_env = ActionMasker(raw_eval_env, get_action_mask)
            eval_env = VecNormalize(
                DummyVecEnv([lambda: Monitor(masked_eval_env)]), **dummy_vec_env_args
            )
            eval_env.obs_rms = train_env.obs_rms
            eval_callback = UnifiedEvalCallback(
                config=config,
                eval_env=eval_env,
                trial=trial,
                checkpoint_dir=checkpoint_dir,
                fold_idx=fold_idx,
                eval_step_offset=eval_step_offset,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose = 1 if trial is None else 0
        seed = (
            (config.data_seed + fold_idx * 100)
            if config.data_seed is not None
            else None
        )
        policy_kwargs = {
            "net_arch": {
                "pi": [config.net_arch_dim, config.net_arch_dim],
                "qf": [config.net_arch_dim, config.net_arch_dim],
            },
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }
        sb3_args_common = {
            "device": device,
            "verbose": verbose,
            "seed": seed,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
        if config.algorithm == "SAC":
            print_if_not_trial(
                logger,
                logging.DEBUG,
                trial,
                f"3. Training SAC model for Fold {fold_idx + 1}...",
            )
            model = SAC(
                env=train_env,
                policy="MlpPolicy",
                ent_coef="auto",
                gamma=config.gamma,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )
        else:
            print_if_not_trial(
                logger,
                logging.DEBUG,
                trial,
                f"3. Training PPO model for Fold {fold_idx + 1}...",
            )
            model = MaskablePPO(
                env=train_env,
                policy="MlpPolicy",
                ent_coef=config.ent_coef_initial,
                clip_range=config.clip_range,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )
        total_training_steps = int(config.timesteps * (1.0 - config.test_fraction))
        entropy_callback = EntropyDecayCallback(
            ent_coef_initial=config.ent_coef_initial,
            ent_coef_final=config.ent_coef_final,
            total_timesteps=total_training_steps,
            verbose=verbose,
        )
        callbacks = [entropy_callback]
        if dashboard_callback:
            callbacks.append(dashboard_callback)
        if eval_callback:
            callbacks.append(eval_callback)
        if callbacks:
            model.learn(total_timesteps=config.timesteps, callback=callbacks)
        else:
            model.learn(total_timesteps=config.timesteps)

        # Pruning cleanup
        if eval_callback:
            eval_env.close()
            if eval_callback.is_pruned:
                train_env.close()
                raise optuna.exceptions.TrialPruned()

        # ----- Testing --------------------------------------------------------------
        print_if_not_trial(
            logger,
            logging.DEBUG,
            trial,
            f"4. Testing trained model for Fold {fold_idx + 1}...",
        )
        shared_env_args_with_logging = shared_env_args.copy()
        shared_env_args_with_logging["config"] = dataclasses.replace(
            shared_env_config, disable_logging=False
        )
        test_env_raw = ActionMasker(
            MinimalCryptoEnv(**shared_env_args_with_logging), get_action_mask
        )
        test_env = VecNormalize(
            DummyVecEnv([lambda: test_env_raw]), **dummy_vec_env_args
        )
        test_env.obs_rms = train_env.obs_rms
        obs = test_env.reset()
        done = False
        eval_steps = 0
        base_test_env: MinimalCryptoEnv = test_env.venv.envs[0].unwrapped
        eval_initial_portfolio_value = base_test_env.portfolio_value
        eval_portfolio_values = [
            {"step": 0, "value": float(eval_initial_portfolio_value)}
        ]
        eval_realized_pnl = [{"step": 0, "value": 0.0}]
        eval_closed_trades = 0
        eval_winning_trades = 0
        eval_reward_totals: dict = {}
        info: dict = {}
        while not done:
            action_masks = np.expand_dims(test_env.venv.envs[0].action_masks(), axis=0)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, _, dones, infos = test_env.step(action)
            done = bool(dones[0])
            info = infos[0]
            if "reward_components" in info:
                for k, v in info["reward_components"].items():
                    eval_reward_totals[k] = eval_reward_totals.get(k, 0) + v
            if info.get("is_valid_sell", False):
                eval_closed_trades += 1
                if info.get("realised_pnl", 0.0) > 0:
                    eval_winning_trades += 1
            eval_steps += 1
            current_pv = info.get(
                "final_portfolio_value", base_test_env.portfolio_value
            )
            eval_portfolio_values.append(
                {"step": eval_steps, "value": float(current_pv)}
            )
            eval_realized_pnl.append(
                {
                    "step": eval_steps,
                    "value": float(current_pv - eval_initial_portfolio_value),
                }
            )
        per_asset_stats = info["per_asset_stats"]
        eval_final_portfolio_value = info.get(
            "final_portfolio_value", base_test_env.portfolio_value
        )
        eval_final_trades_count = info.get(
            "final_trades_count", base_test_env.trades_count
        )
        eval_final_fees_paid = info.get(
            "final_fees_paid", base_test_env.fees_paid_total
        )

        fold_score = calculate_calmar_ratio(eval_portfolio_values)
        fold_scores.append(fold_score)
        if trial is not None:
            ckpt_mgr.save_fold_result(
                trial_number=trial.number,
                fold_idx=fold_idx,
                score=fold_score,
                metrics={"calmar": fold_score},
            )
        eval_win_rate_pct = (
            (eval_winning_trades / eval_closed_trades * 100.0)
            if eval_closed_trades > 0
            else 0.0
        )
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "final_portfolio_value": eval_final_portfolio_value,
                "pnl": float(eval_final_portfolio_value - eval_initial_portfolio_value),
                "calmar": float(fold_score),
                "trades": eval_final_trades_count,
                "closed_trades": int(eval_closed_trades),
                "win_rate_pct": float(eval_win_rate_pct),
                "fees_paid": float(eval_final_fees_paid),
            }
        )

        # Log results --------------------------------------------------------------
        print_if_not_trial(logger, logging.INFO, trial, f"Fold {fold_idx + 1} Results:")
        print_if_not_trial(
            logger,
            logging.INFO,
            trial,
            f"  Final PV: ${eval_final_portfolio_value:.2f} | PnL: ${eval_final_portfolio_value - eval_initial_portfolio_value:.2f} | Calmar: {fold_score:.2f}",
        )
        print_if_not_trial(
            logger,
            logging.INFO,
            trial,
            f"  Trades: {eval_final_trades_count} | Sells: {eval_closed_trades} | Win Rate: {eval_win_rate_pct:.1f}% | Fees: ${eval_final_fees_paid:.4f}",
        )

        last_model = model
        last_eval_reward_totals = eval_reward_totals
        last_eval_portfolio_values = eval_portfolio_values
        last_eval_realized_pnl = eval_realized_pnl
        last_eval_steps = eval_steps

        # Per‑asset breakdown -------------------------------------------------------
        print_if_not_trial(
            logger, logging.INFO, trial, "\nPer-Asset Performance Breakdown:"
        )
        print_if_not_trial(
            logger,
            logging.INFO,
            trial,
            f"{'Symbol':<10} | {'Realized PnL':<13} | {'Total PnL':<11} | {'Trades':<8} | {'Win Rate':<10} | {'Fees':<8}",
        )
        print_if_not_trial(logger, logging.INFO, trial, "-" * 72)
        for sym, stats in per_asset_stats.items():
            print_if_not_trial(
                logger,
                logging.INFO,
                trial,
                f"{sym:<10} | ${stats['realized_pnl']:<12.2f} | ${stats['total_pnl']:<10.2f} | {stats['trades']:<8} | {stats['win_rate_pct']:<9.1f}% | ${stats['fees_paid']:<7.4f}",
            )

        last_test_prices = shared_test_prices
        last_test_static = shared_test_static
        last_test_norm_vol = shared_test_norm_vol
        last_test_names = shared_test_names

        test_env.close()
        last_obs_rms = train_env.obs_rms  # Save before closing; used by multi-seed eval
        train_env.close()

    # ----- Aggregate CV results ---------------------------------------------------
    cv_mean_pv = float(np.mean([r["final_portfolio_value"] for r in fold_results]))
    cv_mean_pnl = float(np.mean([r["pnl"] for r in fold_results]))
    cv_mean_calmar = float(np.mean(fold_scores))
    cv_mean_win_rate = float(np.mean([r["win_rate_pct"] for r in fold_results]))
    cv_total_trades = sum(r["trades"] for r in fold_results)
    cv_total_fees = sum(r["fees_paid"] for r in fold_results)

    return (
        cv_mean_pv,
        cv_mean_pnl,
        cv_mean_calmar,
        cv_mean_win_rate,
        cv_total_trades,
        cv_total_fees,
        fold_results,
        per_asset_stats,
        last_model,
        (
            last_test_prices,
            last_test_static,
            last_test_norm_vol,
            last_test_names,
            last_prices_arr,
            last_asset_names,
        ),
        (
            last_eval_reward_totals,
            last_eval_portfolio_values,
            last_eval_realized_pnl,
            last_eval_steps,
        ),
        last_obs_rms,
    )
