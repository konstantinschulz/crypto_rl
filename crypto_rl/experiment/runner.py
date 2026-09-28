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
from crypto_rl.env.logging_utils import (
    LoggerBase,
    print_if_not_trial,
    selective_logger,
)
from scripts.eval_log_action_counter import eval_log_action_counter
from scripts.eval_report import eval_report

from .dashboard_utils import update_dashboard_state
from .data_loading import load_raw_data, prepare_splits
from .evaluation_utils import compute_buy_and_hold_baseline, run_multi_seed_eval
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
) -> None:
    """Print the walk-forward CV summary to the logger."""
    print_if_not_trial(logger, logging.INFO, trial, "\n" + "=" * 40)
    print_if_not_trial(
        logger, logging.INFO, trial, f"WALK-FORWARD CV SUMMARY ({n_splits} Folds):"
    )
    print_if_not_trial(
        logger, logging.INFO, trial, f"Mean Test Portfolio Value: ${cv_mean_pv:.2f}"
    )
    print_if_not_trial(
        logger, logging.INFO, trial, f"Mean Test PnL:             ${cv_mean_pnl:.2f}"
    )
    print_if_not_trial(
        logger, logging.INFO, trial, f"Mean Test Calmar:          {cv_mean_calmar:.2f}"
    )
    print_if_not_trial(
        logger,
        logging.INFO,
        trial,
        f"Mean Test Win Rate:        {cv_mean_win_rate:.1f}%",
    )
    print_if_not_trial(
        logger, logging.INFO, trial, f"Total CV Trades:           {cv_total_trades}"
    )
    print_if_not_trial(
        logger,
        logging.INFO,
        trial,
        f"Total Test Fees Paid:           ${cv_total_fees:.4f}",
    )
    print_if_not_trial(logger, logging.INFO, trial, "=" * 40 + "\n")


def run_experiment(
    config: RLConfig,
    trial: optuna.trial.Trial | None = None,
) -> float:
    """Run training and evaluation with walk-forward CV.
    Returns the mean test Calmar across folds (or multi-seed portfolio value) for Optuna.
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
        splits_cache = Path(
            f"{config.data_cache_dir}/seed_{config.data_seed}_rows_{config.n_rows}_splits.json"
        )
        splits_cache.parent.mkdir(parents=True, exist_ok=True)
        raw_df = None  # Default to None to save RAM
        if splits_cache.exists():
            with open(splits_cache, "r") as f:
                splits = json.load(f)
            print_if_not_trial(
                logger,
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
            )

            # ── 5. Buy-and-hold baseline ─────────────────────────────────────
            buy_hold_final = compute_buy_and_hold_baseline(
                config, last_asset_names, last_prices_arr
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
                        buy_hold_baseline=buy_hold_final,
                    )
                    print_if_not_trial(
                        logger,
                        logging.DEBUG,
                        trial,
                        f"Dashboard state updated with evaluation results in {state_file}",
                    )
                except Exception as e:
                    print_if_not_trial(
                        logger,
                        logging.DEBUG,
                        trial,
                        f"Error updating dashboard state: {e}",
                    )

            # ── 8. Final reporting (non-trial runs only) ─────────────────────
            if trial is None:
                eval_log_action_counter(logger)
                eval_report(logger)

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
            print_if_not_trial(
                logger,
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
