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
        # Concurrent workers time-slice the GPU: 8 CPU workers train ~1.6×
        # faster in total than 8 CUDA workers (HYPERPARAMETER_RATIONALE.md)
        device="cpu",
        disable_logging=True,
        net_arch_dim=256,
        n_rows=18_000_000,
        timesteps=1_200_000,  # 3000 / 1_200_000
        train_chunks=5,
        val_fraction=0.05,  # use small validation set to speed up Optuna sweeps and reserve more data for training
        # No longer searched (ROADMAP P2-4): SB3's clip, and the rollout and
        # env settings of scripts/run_large.sh. n_envs stays fixed so every
        # trial uses the same compute per step.
        batch_size=128,
        clip_range=0.2,
        max_asset_allocation=0.20,
        n_envs=9,
        n_steps=1024,
        window_size=240,
    )


# The ROADMAP P2-4 starting point. It is enqueued as the study's first trial,
# so every other trial is compared against it.
BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "drawdown_penalty_coef": 0.0,
}


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

    # The search is narrowed to what remains after P1-1 (ROADMAP P2-4): the
    # learning rate, a constant entropy coefficient, the horizon, and the one
    # shaping term the reward ablation showed to change behavior.
    ent_coef = suggest("ent_coef", trial.suggest_float, 1e-3, 3e-2, log=True)
    trial_config: RLConfig = replace(
        base_config,
        drawdown_penalty_coef=suggest(
            "drawdown_penalty_coef", trial.suggest_categorical, [0.0, 0.25, 0.5]
        ),
        ent_coef_final=ent_coef,
        ent_coef_initial=ent_coef,
        # Per 15-minute decision: horizon 1 / (1 − γ) ≈ 12.5 h to 50 h
        gamma=suggest("gamma", trial.suggest_float, 0.98, 0.995),
        learning_rate=suggest(
            "learning_rate", trial.suggest_float, 5e-5, 1e-3, log=True
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
                f"[Optuna] Trial {fresh_t.number} is actively running in another terminal (PID {worker_pid}). Skipping."
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
        # v4: the narrowed PPO search space of ROADMAP P2-4. Older trials are
        # not comparable: v3 trained with the old entropy-decay horizon, v2
        # used Calmar as the objective, and v1 used the old reward.
        study_name="crypto_rl_v4",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    # Run the baseline first; a no-op if a worker already enqueued or ran it
    study.enqueue_trial(BASELINE_PARAMS, skip_if_exists=True)
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
