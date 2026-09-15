"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import optuna

# Automatically add the repository root (one level up from 'scripts/') to Python's path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.config import RLConfig
from main import run_experiment


def objective(trial: optuna.trial.Trial, override_params: dict | None = None):
    if override_params is None:
        override_params = {}
    # Load defaults, but enforce fast execution for the sweep
    base_config: RLConfig = RLConfig(disable_logging=True)

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
        cv_folds=3,
        dashboard=False,
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
        n_rows=400000,  # 10000 / 20000  / 40000
        n_steps=suggest(
            "n_steps", trial.suggest_categorical, [512, 1024, 2048]
        ),  # 256, 2048, 4096
        profit_bonus=suggest("profit_bonus", trial.suggest_float, 0.0, 0.100, step=0.005),
        timesteps=1500000,  # 30000 / 50000 / 100000 / 1200000 / 1500000
        turnover_penalty=suggest("turnover_penalty", trial.suggest_float, 0.010, 0.30),
        turnover_penalty_steps_threshold=suggest(
            "turnover_penalty_steps_threshold", trial.suggest_int, 5, 25
        ),
        window_size=suggest(
            "window_size", trial.suggest_categorical, [30, 60, 120, 240]
        ),
    )

    try:
        # Pass the trial object directly into Python memory!
        score = run_experiment(trial_config, trial=trial)
        return score
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -float("inf")  # bad trial score for failed trials


def resume_running_trials(study: optuna.Study):
    """Finds trials interrupted in the RUNNING state and resumes them explicitly."""
    running_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    ]

    for t in running_trials:
        # t is a FrozenTrial, so t.params contains the exact dictionary of parameters it used
        interrupted_params = t.params
        print(f"\n[Optuna] Resuming interrupted Trial {t.number} with frozen params.")

        # Instantiate a live Trial object to continue the execution
        trial = optuna.trial.Trial(study, t._trial_id)

        try:
            # Pass the frozen params directly to override default suggestions
            score = objective(trial, override_params=interrupted_params)
            study.tell(t.number, score)
        except optuna.exceptions.TrialPruned:
            study.tell(t.number, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:
            print(f"Trial {t.number} failed with error: {e}")
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)


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
    # 2. Calculate remaining trials so we don't overshoot args.n_trials
    completed = len(
        [
            t
            for t in study.trials
            if t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ]
    )
    remaining = max(0, args.n_trials - completed)
    if remaining > 0:
        print(f"\n[Optuna] Starting optimization for {remaining} remaining trials...")
        study.optimize(lambda trial: objective(trial), n_trials=remaining, n_jobs=1)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
