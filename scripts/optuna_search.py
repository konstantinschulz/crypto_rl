"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20
"""

import os

# Force C-math libraries to use only 1 thread per Optuna worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "1"
import tracemalloc

tracemalloc.start()
import argparse
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import optuna

# Automatically add the repository root (one level up from 'scripts/') to Python's path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.config import RLConfig
from main import run_experiment


def get_base_config() -> RLConfig:
    """Returns a base RLConfig with fixed parameters for Optuna sweeps."""
    return RLConfig(
        cv_folds=3,
        dashboard=False,
        data_seed=42,
        disable_logging=True,
        n_rows=18_000_000,
        timesteps=1_500_000,  # 3000 / 1_500_000
    )


def get_optuna_config(
    trial: optuna.Trial, override_params: dict | None = None
) -> RLConfig:
    # Claim ownership of this trial for the current OS process
    trial.set_user_attr("worker_pid", os.getpid())
    if override_params is None:
        override_params = {}
    # Load defaults, but enforce fast execution and a FIXED DATASET for the sweep
    base_config: RLConfig = get_base_config()

    # Helper to explicitly use frozen parameters if resuming,
    # otherwise ask Optuna to suggest new ones.
    def suggest(name, method, *s_args, **s_kwargs):
        if name in override_params:
            return override_params[name]
        return method(name, *s_args, **s_kwargs)

    # this is one central value for empty_buy_penalty, empty_sell_penalty, illegal_buy_penalty, illegal_sell_penalty
    rule_penalty = suggest("rule_penalty", trial.suggest_float, 1e-7, 1e-4, log=True)
    # Let Optuna overwrite specific targets
    trial_config: RLConfig = replace(
        base_config,
        action_dead_zone=suggest(
            "action_dead_zone", trial.suggest_float, 0.35, 0.65, step=0.05
        ),
        batch_size=suggest(
            "batch_size", trial.suggest_categorical, [64, 128, 256]
        ),  # , 512, 1024
        capital_preservation_bonus=suggest(
            "capital_preservation_bonus", trial.suggest_float, 0.0, 0.001, step=0.0001
        ),
        clip_range=suggest("clip", trial.suggest_float, 0.10, 0.40),
        drawdown_penalty_coef=suggest(
            "drawdown_penalty_coef", trial.suggest_float, 0.01, 0.30
        ),
        empty_buy_penalty=rule_penalty,
        empty_sell_penalty=rule_penalty,
        ent_coef_final=suggest(
            "ent_coef_final", trial.suggest_float, 0.0001, 0.01, log=True
        ),
        ent_coef_initial=suggest(
            "ent_coef_initial", trial.suggest_float, 0.03, 0.10, log=True
        ),
        gamma=suggest("gamma", trial.suggest_float, 0.982, 0.995),
        hold_cost_rate=suggest(
            "hold_cost_rate", trial.suggest_float, 1e-7, 1e-4, log=True
        ),
        hold_penalty_threshold=suggest(
            "hold_penalty_threshold", trial.suggest_float, -0.10, -0.01
        ),
        illegal_buy_penalty=rule_penalty,
        illegal_sell_penalty=rule_penalty,
        learning_rate=suggest(
            "learning_rate", trial.suggest_float, 5e-6, 5e-5, log=True
        ),
        loss_cut_bonus=suggest("loss_cut_bonus", trial.suggest_float, 0.0, 0.005),
        max_asset_allocation=suggest(
            "max_asset_allocation", trial.suggest_float, 0.10, 0.50, step=0.05
        ),
        max_single_step_allocation=suggest(
            "max_single_step_allocation", trial.suggest_float, 0.30, 0.70, step=0.05
        ),
        n_envs=suggest("n_envs", trial.suggest_int, 6, 12),
        n_steps=suggest(
            "n_steps", trial.suggest_categorical, [512, 1024, 2048]
        ),  # 256, 2048, 4096
        profit_bonus=suggest(
            "profit_bonus", trial.suggest_float, 0.0, 0.100, step=0.005
        ),
        turnover_penalty=suggest("turnover_penalty", trial.suggest_float, 0.010, 0.30),
        turnover_penalty_steps_threshold=suggest(
            "turnover_penalty_steps_threshold", trial.suggest_int, 5, 25
        ),
        window_size=suggest(
            "window_size", trial.suggest_categorical, [30, 60, 120, 240]
        ),
    )
    return trial_config


def is_pid_alive(pid: int) -> bool:
    """Cross-platform check to see if a specific Process ID is currently running."""
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback for Linux/Ubuntu (works natively without psutil)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def objective(trial: optuna.trial.Trial, override_params: dict | None = None):
    """
    Objective function for Optuna hyper-parameter optimization.
    This function defines the search space and runs the experiment with suggested parameters.
    """
    trial_config: RLConfig = get_optuna_config(trial, override_params)
    try:
        # Pass the trial object directly into Python memory!
        score = run_experiment(trial_config, trial=trial)
        return score
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        print(traceback.format_exc())
        return -float("inf")  # bad trial score for failed trials


def resume_running_trials(study: optuna.Study):
    """Finds trials interrupted in the RUNNING state and resumes them explicitly."""
    # Get the initial stale list of running trials at boot
    initial_running_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    ]

    for stale_t in initial_running_trials:
        # --- NEW: Re-fetch the trial directly from the DB to get absolute latest PIDs and states ---
        fresh_t: optuna.trial.FrozenTrial = next(
            x for x in study.get_trials() if x.number == stale_t.number
        )

        # If another worker already finished or pruned this trial while we were busy, skip it safely
        if fresh_t.state != optuna.trial.TrialState.RUNNING:
            continue

        worker_pid = fresh_t.user_attrs.get("worker_pid")

        # Check if another terminal is actively processing this trial right now
        if worker_pid is not None and is_pid_alive(worker_pid):
            print(
                f"\n[Optuna] Trial {fresh_t.number} is actively running in another terminal (PID {worker_pid}). Skipping."
            )
            continue

        # t is a FrozenTrial, so fresh_t.params contains the exact dictionary of parameters it used
        interrupted_params = fresh_t.params
        print(
            f"\n[Optuna] Resuming interrupted Trial {fresh_t.number} with frozen params."
        )

        # Instantiate a live Trial object to continue the execution
        trial = optuna.trial.Trial(study, fresh_t._trial_id)
        try:
            # Pass the frozen params directly to override default suggestions
            score = objective(trial, override_params=interrupted_params)
            study.tell(fresh_t.number, score)
            print(
                f"\n[{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [Optuna] ✅ Resumed Trial {fresh_t.number} finished with value: {score}"
            )
            print(f"[Optuna] Parameters used: {interrupted_params}")
        except optuna.exceptions.TrialPruned:
            study.tell(fresh_t.number, state=optuna.trial.TrialState.PRUNED)
            print(f"\n[Optuna] ✂️ Resumed Trial {fresh_t.number} was PRUNED.")
        except Exception as e:
            print(f"Trial {fresh_t.number} failed with error: {e}")
            study.tell(fresh_t.number, state=optuna.trial.TrialState.FAIL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name="crypto_rl_v1",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    # 1. Manually resolve any interrupted trials from a previous session
    resume_running_trials(study)
    completed_statuses = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    )
    # 2. Calculate remaining trials so we don't overshoot args.n_trials
    completed = len([t for t in study.trials if t.state in completed_statuses])
    remaining = max(0, args.n_trials - completed)
    if remaining > 0:
        print(f"\n[Optuna] Starting optimization for {remaining} remaining trials...")
        study.optimize(lambda trial: objective(trial), n_trials=remaining, n_jobs=1)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
