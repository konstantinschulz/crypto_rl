"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20 --n-rows 10000 --timesteps 30000
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


def objective(trial: optuna.trial.Trial):
    # Load defaults, but enforce fast execution for the sweep
    base_config: RLConfig = RLConfig(disable_logging=True)
    # this is one central value for empty_buy_penalty, empty_sell_penalty, illegal_buy_penalty, illegal_sell_penalty
    rule_penalty = trial.suggest_float("rule_penalty", 1e-6, 1e-3, log=True)
    # Let Optuna overwrite specific targets
    trial_config: RLConfig = replace(
        base_config,
        batch_size=trial.suggest_categorical(
            "batch_size", [64, 128, 256]
        ),  # , 512, 1024
        clip_range=trial.suggest_float("clip", 0.10, 0.30),
        dashboard=False,
        drawdown_penalty_coef=trial.suggest_float("drawdown_penalty_coef", 0.2, 0.8),
        empty_buy_penalty=rule_penalty,
        empty_sell_penalty=rule_penalty,
        ent_coef=trial.suggest_float("ent_coef", 0.008, 0.03, log=True),
        gamma=trial.suggest_float("gamma", 0.985, 0.998),
        illegal_buy_penalty=rule_penalty,
        illegal_sell_penalty=rule_penalty,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        min_turnover_threshold=trial.suggest_float("min_turnover_threshold", 0.05, 0.15),
        n_envs=trial.suggest_int("n_envs", 5, 10),
        n_rows=400000,  # 10000 / 20000  / 40000
        n_steps=trial.suggest_categorical("n_steps", [512, 1024]),  # 256, 2048, 4096
        profit_bonus=trial.suggest_float("profit_bonus", 0.0, 0.01),
        timesteps=1200000,  # 30000 / 50000 / 100000
        window_size=trial.suggest_categorical("window_size", [30, 60, 120, 240]),
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
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
