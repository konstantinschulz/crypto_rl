# -*- coding: utf-8 -*-
"""Helper functions to run folds for the experiment.

This module contains the heavy‑weight loop that was previously inside
`run_experiment`.  Keeping the logic here keeps `runner.py` concise while
preserving the original behavior.
"""

import dataclasses
import gc
import json
import logging
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from crypto_rl.callbacks import (
    DashboardCallback,
    EntropyDecayCallback,
    UnifiedEvalCallback,
)
from crypto_rl.checkpoint_manager import CVCheckpointManager
from crypto_rl.config import RLConfig
from crypto_rl.data import DEFAULT_SYMBOLS, FoldSplit
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.data_utils import (
    build_fold_caches,
    compute_static_obs_from_long_df,
    fold_cache_prefixes,
    is_cache_built,
    parquet_open_time_range,
)
from crypto_rl.env.logging_utils import (
    LoggerBase,
    send_notification,
)
from crypto_rl.env.metrics import calculate_calmar_ratio
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.data_loading import load_raw_data
from crypto_rl.experiment.evaluation_utils import (
    compute_window_baselines,
    excess_returns,
    log_baselines,
)


def build_policy_kwargs(config: RLConfig) -> dict[str, Any]:
    """Policy and value network sizes for MaskablePPO.

    SB3 reads the value network from the ``"vf"`` key. A ``"qf"`` key (SAC
    terminology) is silently ignored, which leaves the critic as a single
    linear layer.
    """
    hidden = [config.net_arch_dim, config.net_arch_dim]
    return {
        "net_arch": {"pi": list(hidden), "vf": list(hidden)},
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
    }


def _eval_env_args(
    arrays: tuple, *, run_id: str, is_fast_eval: bool, config: RLConfig, logger: LoggerBase
) -> dict[str, Any]:
    """Keyword arguments for an evaluation `MinimalCryptoEnv` over one slice."""
    prices, static, norm_vol, names = arrays
    return {
        "prices_arr": prices,
        "static_obs": static,
        "norm_vol_arr": norm_vol,
        "asset_names": names,
        "run_id": run_id,
        "is_eval": True,
        "is_fast_eval": is_fast_eval,
        "config": config,
        "logger": logger,
    }


def _run_episode(
    model: MaskablePPO,
    env_args: dict[str, Any],
    obs_rms: Any,
    dummy_vec_env_args: dict[str, Any],
    desc: str,
    show_progress: bool,
) -> dict[str, Any]:
    """Run one deterministic episode of `model` and collect its results."""
    masked_env = ActionMasker(MinimalCryptoEnv(**env_args), get_action_mask)
    vec_env = VecNormalize(DummyVecEnv([lambda: masked_env]), **dummy_vec_env_args)
    vec_env.obs_rms = obs_rms
    obs = vec_env.reset()
    base_env: MinimalCryptoEnv = vec_env.venv.envs[0].unwrapped
    initial_pv = base_env.portfolio_value
    portfolio_values = [{"step": 0, "value": float(initial_pv)}]
    realized_pnl = [{"step": 0, "value": 0.0}]
    reward_totals: dict = {}
    closed_trades = 0
    winning_trades = 0
    steps = 0
    info: dict = {}
    done = False
    # Calculate exactly how many steps this evaluation will take
    total_steps = base_env.max_steps - base_env.current_step
    # Passive baselines over exactly the steps this episode is scored on
    baselines = compute_window_baselines(
        base_env.prices_arr,
        start_step=base_env.current_step,
        max_steps=base_env.max_steps,
        asset_names=base_env.asset_names,
        fee_rate=base_env.fee_rate,
        budget_initial=initial_pv,
    )
    # Progress bars are disabled for headless background Optuna workers
    with tqdm(
        total=total_steps, desc=desc, disable=not show_progress, unit="steps"
    ) as pbar:
        while not done:
            action_masks = np.expand_dims(vec_env.venv.envs[0].action_masks(), axis=0)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            info = infos[0]
            for k, v in info.get("reward_components", {}).items():
                reward_totals[k] = reward_totals.get(k, 0) + v
            if info.get("is_valid_sell", False):
                closed_trades += 1
                if info.get("realised_pnl", 0.0) > 0:
                    winning_trades += 1
            steps += 1
            pbar.update(1)
            current_pv = info.get("final_portfolio_value", base_env.portfolio_value)
            portfolio_values.append({"step": steps, "value": float(current_pv)})
            realized_pnl.append(
                {"step": steps, "value": float(current_pv - initial_pv)}
            )

    final_pv = info.get("final_portfolio_value", base_env.portfolio_value)
    result = {
        "initial_portfolio_value": initial_pv,
        "final_portfolio_value": final_pv,
        "baselines": baselines,
        "excess_returns": excess_returns(final_pv, baselines, initial_pv),
        "trades_count": info.get("final_trades_count", base_env.trades_count),
        "fees_paid": info.get("final_fees_paid", base_env.fees_paid_total),
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "steps": steps,
        "portfolio_values": portfolio_values,
        "realized_pnl": realized_pnl,
        "reward_totals": reward_totals,
        "per_asset_stats": info["per_asset_stats"],
        "action_type_counts": info.get(
            "action_type_counts", base_env.action_type_counts
        ),
        "traded_coins": sorted(info.get("traded_coins", base_env.traded_coins)),
        "env_portfolio_values": info.get(
            "portfolio_values", base_env.portfolio_values
        ),
        "calmar": calculate_calmar_ratio(portfolio_values),
    }
    vec_env.close()
    return result


def run_folds(
    config: RLConfig,
    trial: optuna.trial.Trial | None,
    logger: LoggerBase,
    run_id: str,
    run_dir: Path,
    state_file: Path,
    index_file: Path,
    ckpt_mgr: CVCheckpointManager,
    env_config: RLConfig,
    dummy_vec_env_args: dict[str, Any],
    splits: list[FoldSplit],
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
    evals_per_fold = config.timesteps // int(config.eval_freq)
    # 1. Fetch completed folds if resuming a specific trial
    completed_folds: dict = {}
    if trial is not None:
        existing_ckpt = ckpt_mgr.load_checkpoint(trial.number)
        if existing_ckpt:
            completed_folds = existing_ckpt.get("completed_folds", {})
    # =========================================================================
    # PHASE 1: THE PRE-FLIGHT CACHE BUILDER
    # Pre-build all required memory-mapped caches so we can completely destroy
    # raw_df before the heavy PyTorch neural networks spin up.
    # =========================================================================
    num_assets = len(DEFAULT_SYMBOLS)
    data_range = parquet_open_time_range(config.parquet_path)
    # Splits may come from the JSON cache as plain lists
    splits = [FoldSplit(*split) for split in splits]
    # The test slice is only touched by the final report (non-trial runs), so
    # Optuna trials never build or read it.
    cache_slices = ("train", "val") if trial is not None else ("train", "val", "test")
    fold_prefixes = [
        fold_cache_prefixes(config, num_assets, fold_idx, data_range, split)
        for fold_idx, split in enumerate(splits)
    ]
    needs_raw_df = any(
        not is_cache_built(config, getattr(prefixes, name))
        for prefixes in fold_prefixes
        for name in cache_slices
    )
    if needs_raw_df:
        logger.print_if_not_trial(
            logging.DEBUG,
            trial,
            "Pre-flight: Loading raw data to build missing caches...",
        )
        raw_df = load_raw_data(config, logger, trial)
        for split, prefixes in zip(splits, fold_prefixes):
            build_fold_caches(raw_df, config, split, prefixes, cache_slices)
        del raw_df
        gc.collect()
        # --- NEW: FORCE C-ALLOCATOR TO RETURN RAM TO LINUX ---
        import ctypes

        try:
            # malloc_trim(0) forces glibc to release free memory back to the OS
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print(f"[PID {os.getpid()}] glibc malloc_trim executed successfully.")
        except Exception as e:
            pass
    # =========================================================================
    # PHASE 2: EXECUTE FOLDS (TRAINING & EVALUATION)
    # =========================================================================
    for fold_idx, split in enumerate(splits):
        training_start_str, training_end_str = split.train_start_str, split.train_end_str
        prefixes = fold_prefixes[fold_idx]

        # We know the cache exists now, so we pass None safely!
        prices_arr, static_obs, norm_vol_arr, asset_names = (
            compute_static_obs_from_long_df(None, config, cache_prefix=prefixes.train)
        )
        str_fold = str(fold_idx)
        # ----- Resume from checkpoint ------------------------------------------------
        if str_fold in completed_folds:
            cached_score = completed_folds[str_fold]["score"]
            logger.print_if_not_trial(
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
        if trial is not None:
            msg = f"[PID {os.getpid()}] Optuna Trial {trial.number} " + msg
        logger.print_if_not_trial(logging.DEBUG, None, msg)
        send_notification(
            msg,
            summary="Experiment Run"
            if trial is None
            else f"Optuna Trial {trial.number}",
        )
        eval_step_offset = fold_idx * evals_per_fold
        last_prices_arr = prices_arr
        last_asset_names = asset_names
        logger.print_if_not_trial(logging.DEBUG, trial, "2. Setting up environment...")

        def make_env():
            e = MinimalCryptoEnv(
                config=env_config,
                prices_arr=prices_arr,
                static_obs=static_obs,
                norm_vol_arr=norm_vol_arr,
                asset_names=asset_names,
                run_id=run_id,
                logger=logger,
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

        # ----- Validation observations ----------------------------------------------
        # Validation drives intermediate evals, checkpoint ranking and pruning.
        # The test slice is not loaded here.
        logger.print_if_not_trial(
            logging.DEBUG, trial, "Computing validation observations..."
        )
        val_arrays = compute_static_obs_from_long_df(
            None, config, cache_prefix=prefixes.val
        )
        checkpoint_dir = run_dir / f"checkpoints_fold_{fold_idx + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        val_env_args = _eval_env_args(
            val_arrays,
            run_id=run_id,
            # Short slice for the many intermediate evals during training
            is_fast_eval=True,
            # Logging stays off: only the final test run writes the eval log
            config=dataclasses.replace(env_config, disable_logging=True),
            logger=logger,
        )
        eval_callback = None
        eval_env = None
        if config.checkpoint or trial is not None:
            raw_eval_env = MinimalCryptoEnv(**val_env_args)
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
        policy_kwargs = build_policy_kwargs(config)
        sb3_args_common = {
            "device": device,
            "verbose": verbose,
            "seed": seed,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
        if config.algorithm != "PPO":
            raise NotImplementedError(
                f"Algorithm {config.algorithm} is not supported in this runner."
            )
        logger.print_if_not_trial(
            logging.DEBUG,
            trial,
            f"3. Training PPO model for Fold {fold_idx + 1}...",
        )
        # =====================================================================
        # MID-FOLD CHUNKING & RESUMPTION LOGIC
        # =====================================================================
        # 1. Define a persistent directory that survives Ctrl+C (bypassing timestamped run_id)
        sub_dir: str = "single_run" if trial is None else f"trial_{trial.number}"
        chunk_dir = os.path.join(config.chunks_dir, sub_dir)
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_model_path = os.path.join(chunk_dir, f"fold_{fold_idx}_model.zip")
        chunk_buffer_path = os.path.join(chunk_dir, f"fold_{fold_idx}_buffer.pkl")
        chunk_meta_path = os.path.join(chunk_dir, f"fold_{fold_idx}_meta.json")
        # Break the fold into smaller, frequent save points
        chunk_steps = config.timesteps // config.train_chunks
        start_chunk = 0
        model: MaskablePPO

        # 2. Attempt to load an existing chunk from a previous interrupted run
        if os.path.exists(chunk_meta_path):
            try:
                with open(chunk_meta_path, "r") as f:
                    start_chunk = json.load(f).get("chunk", 0)
                logger.debug(
                    f"Resuming Fold {fold_idx + 1} from Chunk {start_chunk}/{config.train_chunks}...",
                )
                model = MaskablePPO.load(chunk_model_path, env=train_env)
                # CRITICAL: SB3 resets timesteps on load. We must restore it so
                # your EntropyDecayCallback and UnifiedEvalCallback don't break!
                model.num_timesteps = start_chunk * chunk_steps
            except Exception as e:
                logger.debug(f"Failed to load chunk: {e}. Starting fresh.")
                start_chunk = 0
        # 3. Instantiate fresh model if no chunk exists
        if start_chunk == 0:
            model = MaskablePPO(
                env=train_env,
                policy="MlpPolicy",
                ent_coef=config.ent_coef_initial,
                clip_range=config.clip_range,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )
        # 4. Setup Callbacks
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
        # 5. Execute the Chunked Training Loop
        for chunk in range(start_chunk, config.train_chunks):
            # Prevent drift on the last chunk due to integer division
            steps_this_chunk = (
                chunk_steps
                if chunk < config.train_chunks - 1
                else config.timesteps - (chunk * chunk_steps)
            )
            # Learn seamlessly
            model.learn(
                total_timesteps=steps_this_chunk,
                callback=callbacks,
                reset_num_timesteps=False,  # Tells SB3 to continue exactly where it left off
            )
            # Stop chunking immediately if Optuna decided to prune the trial
            if eval_callback and eval_callback.is_pruned:
                break
            # Save chunk state mid-fold
            model.save(chunk_model_path)
            with open(chunk_meta_path, "w") as f:
                json.dump({"chunk": chunk + 1}, f)
        # 6. Cleanup when fold finishes successfully to save SSD space
        for path in [chunk_model_path, chunk_buffer_path, chunk_meta_path]:
            if os.path.exists(path):
                os.remove(path)
        # =====================================================================
        # Pruning cleanup
        if eval_callback:
            eval_env.close()
            if eval_callback.is_pruned:
                train_env.close()
                raise optuna.exceptions.TrialPruned()

        # ----- Final evaluation -----------------------------------------------------
        # Load the checkpoint that scored best on validation. Its observation
        # statistics were saved with it; the live train_env.obs_rms has drifted.
        best_path = eval_callback.best_model_path if eval_callback else None
        if best_path is not None:
            logger.print_if_not_trial(
                logging.DEBUG,
                trial,
                f"4. Loading best validation checkpoint for Fold {fold_idx + 1} "
                f"(val Calmar {eval_callback.best_calmar:.4f})...",
            )
            eval_model = MaskablePPO.load(best_path, device=device)
            eval_obs_rms = eval_callback.best_obs_rms
        else:
            logger.print_if_not_trial(
                logging.WARNING,
                trial,
                f"No validation checkpoint exists for Fold {fold_idx + 1} (no "
                "intermediate eval ran); using the final model.",
            )
            eval_model = model
            eval_obs_rms = train_env.obs_rms

        if trial is not None:
            # Full-length validation episode: the Optuna objective. Trials
            # never touch the test slice.
            result = _run_episode(
                eval_model,
                {**val_env_args, "is_fast_eval": False},
                eval_obs_rms,
                dummy_vec_env_args,
                desc=f"Validating Fold {fold_idx + 1}",
                show_progress=False,
            )
            reported_arrays = val_arrays
            reported_split = "val"
        else:
            # The test slice is touched exactly once, for the final report,
            # and always over the whole test fold (is_fast_eval=False).
            logger.print_if_not_trial(
                logging.DEBUG,
                trial,
                f"5. Testing best checkpoint on Fold {fold_idx + 1}...",
            )
            test_arrays = compute_static_obs_from_long_df(
                None, config, cache_prefix=prefixes.test
            )
            result = _run_episode(
                eval_model,
                _eval_env_args(
                    test_arrays,
                    run_id=run_id,
                    is_fast_eval=False,
                    config=dataclasses.replace(env_config, disable_logging=False),
                    logger=logger,
                ),
                eval_obs_rms,
                dummy_vec_env_args,
                desc=f"Testing Fold {fold_idx + 1}",
                show_progress=True,
            )
            reported_arrays = test_arrays
            reported_split = "test"

        per_asset_stats = result["per_asset_stats"]
        eval_initial_portfolio_value = result["initial_portfolio_value"]
        eval_final_portfolio_value = result["final_portfolio_value"]
        eval_final_trades_count = result["trades_count"]
        eval_final_fees_paid = result["fees_paid"]
        eval_closed_trades = result["closed_trades"]
        fold_score = result["calmar"]
        fold_scores.append(fold_score)
        if trial is not None:
            ckpt_mgr.save_fold_result(
                trial_number=trial.number,
                fold_idx=fold_idx,
                score=fold_score,
                metrics={"calmar": fold_score},
            )
        eval_win_rate_pct = (
            (result["winning_trades"] / eval_closed_trades * 100.0)
            if eval_closed_trades > 0
            else 0.0
        )
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "split": reported_split,
                "final_portfolio_value": eval_final_portfolio_value,
                "pnl": float(eval_final_portfolio_value - eval_initial_portfolio_value),
                "calmar": float(fold_score),
                "trades": eval_final_trades_count,
                "closed_trades": int(eval_closed_trades),
                "win_rate_pct": float(eval_win_rate_pct),
                "fees_paid": float(eval_final_fees_paid),
                "baselines": result["baselines"],
                "excess_returns": result["excess_returns"],
                "action_type_counts": result["action_type_counts"],
                "traded_coins": result["traded_coins"],
                "portfolio_values": result["env_portfolio_values"],
            }
        )

        # Log results --------------------------------------------------------------
        logger.print_if_not_trial(
            logging.INFO,
            trial,
            f"Fold {fold_idx + 1} Results ({reported_split}, best-validation checkpoint):",
        )
        logger.print_if_not_trial(
            logging.INFO,
            trial,
            f"  Final PV: ${eval_final_portfolio_value:.2f} | PnL: ${eval_final_portfolio_value - eval_initial_portfolio_value:.2f} | Calmar: {fold_score:.2f}",
        )
        logger.print_if_not_trial(
            logging.INFO,
            trial,
            f"  Trades: {eval_final_trades_count} | Sells: {eval_closed_trades} | Win Rate: {eval_win_rate_pct:.1f}% | Fees: ${eval_final_fees_paid:.4f}",
        )
        log_baselines(
            logger,
            trial,
            f"Baselines ({reported_split}, same {result['steps']} steps, same fees):",
            bot_return=(eval_final_portfolio_value - eval_initial_portfolio_value)
            / eval_initial_portfolio_value,
            baselines=result["baselines"],
            excess=result["excess_returns"],
            budget_initial=eval_initial_portfolio_value,
        )

        last_model = eval_model
        last_eval_reward_totals = result["reward_totals"]
        last_eval_portfolio_values = result["portfolio_values"]
        last_eval_realized_pnl = result["realized_pnl"]
        last_eval_steps = result["steps"]
        # Per‑asset breakdown -------------------------------------------------------
        logger.log("\nPer-Asset Performance Breakdown:", trial)
        logger.log(
            f"{'Symbol':<10} | {'Realized PnL':<13} | {'Total PnL':<11} | {'Trades':<8} | {'Win Rate':<10} | {'Fees':<8}",
            trial,
        )
        logger.log("-" * 72, trial)
        for sym, stats in per_asset_stats.items():
            logger.log(
                f"{sym:<10} | ${stats['realized_pnl']:<12.2f} | ${stats['total_pnl']:<10.2f} | {stats['trades']:<8} | {stats['win_rate_pct']:<9.1f}% | ${stats['fees_paid']:<7.4f}",
                trial,
            )
        # Arrays of the split that was reported (val in trials, test otherwise);
        # the baseline and multi-seed evaluation must use the same window.
        (
            last_test_prices,
            last_test_static,
            last_test_norm_vol,
            last_test_names,
        ) = reported_arrays
        last_obs_rms = eval_obs_rms  # Used by multi-seed eval; matches eval_model
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
