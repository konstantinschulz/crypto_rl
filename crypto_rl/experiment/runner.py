import dataclasses
import gc
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from crypto_rl.checkpoint_manager import CVCheckpointManager
from crypto_rl.config import RLConfig
from crypto_rl.env.data_utils import parquet_open_time_range, split_cache_name
from crypto_rl.env.logging_utils import (
    LoggerBase,
    selective_logger,
)
from eval_report import eval_report

from .dashboard_utils import update_dashboard_state
from .data_loading import load_raw_data, prepare_splits
from .evaluation_utils import (
    BASELINE_NAMES,
    log_baselines,
    run_multi_seed_eval,
)
from .fold_runner import run_folds

CLIP_OBS = 10.0  # Clamps normalized inputs to [-10.0, 10.0]
dummy_vec_env_args: dict[str, Any] = {
    "norm_reward": False,
    "norm_obs": True,
    "clip_obs": CLIP_OBS,
    "training": False,
}


def _print_cv_summary(
    logger: LoggerBase,
    trial: optuna.trial.Trial | None,
    n_splits: int,
    cv_mean_pv: float,
    cv_mean_pnl: float,
    cv_mean_calmar: float,
    cv_mean_win_rate: float,
    cv_total_trades: int,
    cv_total_fees: float,
    split_name: str,
) -> None:
    """Print the walk-forward CV summary to the logger.

    `split_name` is the slice the numbers come from: "Validation" for Optuna
    trials (which never touch the test slice), "Test" for the final report.
    """
    logger.log("\n" + "=" * 40, trial)
    logger.log(f"WALK-FORWARD CV SUMMARY ({n_splits} Folds):", trial)
    logger.log(f"Mean {split_name} Portfolio Value: ${cv_mean_pv:.2f}", trial)
    logger.log(f"Mean {split_name} PnL:             ${cv_mean_pnl:.2f}", trial)
    logger.log(f"Mean {split_name} Calmar:          {cv_mean_calmar:.2f}", trial)
    logger.log(f"Mean {split_name} Win Rate:        {cv_mean_win_rate:.1f}%", trial)
    logger.log(f"Total CV Trades:           {cv_total_trades}", trial)
    logger.log(f"Total {split_name} Fees Paid:           ${cv_total_fees:.4f}", trial)
    logger.log("=" * 40 + "\n", trial)


def _mean_over_folds(fold_results: list[dict], key: str) -> dict[str, float]:
    """Per-baseline mean of `fold_results[i][key]`; empty if no fold has it."""
    rows = [r[key] for r in fold_results if key in r]
    if not rows:
        return {}
    return {name: float(np.mean([row[name] for row in rows])) for name in BASELINE_NAMES}


def run_experiment(
    config: RLConfig,
    trial: optuna.trial.Trial | None = None,
) -> float:
    """Run training and evaluation with walk-forward CV.

    Each fold is split into train / validation / test. Validation drives the
    intermediate evals, checkpoint ranking and pruning; the best-on-validation
    checkpoint is evaluated afterwards. Optuna trials return the mean
    *validation* Calmar and never touch the test slice. Non-trial runs also
    evaluate that checkpoint once on the full test fold for the final report.
    """
    # Seed for reproducibility of dataset split
    np.random.seed(config.data_seed)
    log_file_name: str = "latest_results.log"
    if trial is not None:
        log_file_name = os.path.join(
            config.optuna_dir, f"optuna_trial_{trial.number}.log"
        )
    with selective_logger(log_file_name) as logger:
        ckpt_mgr: CVCheckpointManager = CVCheckpointManager(logger)
        if config.eval_freq == "auto":
            config.eval_freq = max(2000, config.timesteps // 10)
        # ── 1. Load data & create walk-forward splits ────────────────────────
        data_range = parquet_open_time_range(config.parquet_path)
        splits_cache = Path(config.data_cache_dir) / split_cache_name(
            config, data_range
        )
        splits_cache.parent.mkdir(parents=True, exist_ok=True)
        raw_df = None  # Default to None to save RAM
        if splits_cache.exists():
            with open(splits_cache, "r") as f:
                splits = json.load(f)
            logger.print_if_not_trial(
                logging.DEBUG,
                trial,
                "1. Loaded walk-forward splits from cache.",
            )
        else:
            raw_df_tmp = load_raw_data(config, logger, trial)
            splits = prepare_splits(raw_df_tmp, config, logger, trial)
            # Save splits atomically for the next workers
            tmp_json = splits_cache.with_suffix(".tmp")
            with open(tmp_json, "w") as f:
                json.dump(splits, f)
            tmp_json.rename(splits_cache)
            # DESTROY THE CALLER REFERENCE BEFORE LAUNCHING RUN_FOLDS
            del raw_df_tmp
            gc.collect()
        n_splits = len(splits)
        # ── 2. Set up run directory ──────────────────────────────────────────
        run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")
        run_dir = Path(config.base_run_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"
        index_file = Path("rl_dashboard_index.json")

        env_config = dataclasses.replace(
            config, disable_logging=trial is not None, parquet_path=None, n_rows=0
        )

        try:
            # ── 3. Execute all CV folds ──────────────────────────────────────
            (
                cv_mean_pv,
                cv_mean_pnl,
                cv_mean_calmar,
                cv_mean_win_rate,
                cv_total_trades,
                cv_total_fees,
                fold_results,
                per_asset_stats,
                last_model,
                test_data_tuple,
                eval_data_tuple,
                last_obs_rms,
            ) = run_folds(
                config=config,
                trial=trial,
                logger=logger,
                run_id=run_id,
                run_dir=run_dir,
                state_file=state_file,
                index_file=index_file,
                ckpt_mgr=ckpt_mgr,
                env_config=env_config,
                dummy_vec_env_args=dummy_vec_env_args,
                splits=splits,
            )
            (
                last_test_prices,
                last_test_static,
                last_test_norm_vol,
                last_test_names,
                last_prices_arr,
                last_asset_names,
            ) = test_data_tuple

            (
                last_eval_reward_totals,
                last_eval_portfolio_values,
                last_eval_realized_pnl,
                last_eval_steps,
            ) = eval_data_tuple

            # ── 4. Print CV summary ──────────────────────────────────────────
            _print_cv_summary(
                logger,
                trial,
                n_splits,
                cv_mean_pv,
                cv_mean_pnl,
                cv_mean_calmar,
                cv_mean_win_rate,
                cv_total_trades,
                cv_total_fees,
                split_name="Validation" if trial is not None else "Test",
            )

            # ── 5. Baselines over the same windows the bot was scored on ─────
            cv_baselines = _mean_over_folds(fold_results, "baselines")
            cv_excess = _mean_over_folds(fold_results, "excess_returns")
            if cv_baselines:
                log_baselines(
                    logger,
                    trial,
                    f"Mean baselines over {len(fold_results)} fold(s), same "
                    "windows and fees as the bot:",
                    bot_return=cv_mean_pnl / config.budget_initial,
                    baselines=cv_baselines,
                    excess=cv_excess,
                    budget_initial=config.budget_initial,
                )
            # ── 6. Multi-seed evaluation ─────────────────────────────────────
            run_multi_seed_eval(
                config=config,
                logger=logger,
                trial=trial,
                last_model=last_model,
                last_test_prices=last_test_prices,
                last_test_static=last_test_static,
                last_test_norm_vol=last_test_norm_vol,
                last_test_names=last_test_names,
                run_id=run_id,
                train_env_obs_rms=last_obs_rms,
                dummy_vec_env_args=dummy_vec_env_args,
            )

            # ── 7. Dashboard update ──────────────────────────────────────────
            if config.dashboard and state_file.exists():
                try:
                    update_dashboard_state(
                        state_file=state_file,
                        config=config,
                        cv_mean_pv=cv_mean_pv,
                        cv_mean_pnl=cv_mean_pnl,
                        cv_mean_calmar=cv_mean_calmar,
                        cv_mean_win_rate=cv_mean_win_rate,
                        cv_total_trades=cv_total_trades,
                        cv_total_fees=cv_total_fees,
                        fold_results=fold_results,
                        per_asset_stats=per_asset_stats,
                        last_eval_reward_totals=last_eval_reward_totals,
                        last_eval_steps=last_eval_steps,
                        last_eval_portfolio_values=last_eval_portfolio_values,
                        last_eval_realized_pnl=last_eval_realized_pnl,
                        baselines=cv_baselines,
                        excess_returns=cv_excess,
                    )
                    logger.print_if_not_trial(
                        logging.DEBUG,
                        trial,
                        f"Dashboard state updated with evaluation results in {state_file}",
                    )
                except Exception as e:
                    logger.print_if_not_trial(
                        logging.DEBUG,
                        trial,
                        f"Error updating dashboard state: {e}",
                    )
            logger.print_if_not_trial(
                logging.DEBUG, trial, "Results from the final evaluation:"
            )
            # ── 8. Final reporting (non-trial runs only) ─────────────────────
            eval_report(
                logger,
                config,
                trial,
                fold_results,
                cv_mean_pv,
                cv_mean_calmar,
                state_file=state_file,
            )

            # Trial finished all folds successfully: clean up checkpoint file
            if trial is not None:
                ckpt_mgr.clear_checkpoint(trial.number)

            return float(cv_mean_calmar)  # Return mean Calmar ratio for Optuna

        except optuna.exceptions.TrialPruned:
            assert trial is not None, (
                "TrialPruned should only occur during an Optuna trial."
            )
            # Clean up checkpoint and let Optuna handle the pruned trial
            ckpt_mgr.clear_checkpoint(trial.number)
            raise
        except KeyboardInterrupt:
            # Prevent Optuna from catching this and marking the trial as FAIL.
            # Hard-exit preserves the "RUNNING" state in the DB for later resumption.
            logger.print_if_not_trial(
                logging.DEBUG,
                None,
                "\n[!] Ctrl+C (KeyboardInterrupt) detected. Hard-exiting to preserve "
                "trial in RUNNING state for later resumption...",
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        except Exception as e:
            # Leave checkpoint intact on crash so the trial can resume later
            raise e
