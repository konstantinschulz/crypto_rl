"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20 --n-rows 10000 --timesteps 30000
"""

import argparse
import sys
from pathlib import Path

import optuna

# Automatically add the repository root (one level up from 'scripts/') to Python's path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.cli import build_parser
from main import run_experiment


def objective(trial: optuna.trial.Trial):
    parser = build_parser()
    # Parse default args or mock them into a Namespace
    args = parser.parse_args([])

    # Suggest hyperparameters via Optuna
    args.action_dead_zone = trial.suggest_float("action_dead_zone", 0.05, 0.30)
    args.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    args.clip_range = trial.suggest_float("clip", 0.01, 0.3)
    args.drawdown_penalty_coef = trial.suggest_float("drawdown_penalty_coef", 5, 20)
    # args.empty_buy_penalty = trial.suggest_float("empty_buy_penalty", 1e-4, 1e-1)
    args.empty_sell_penalty = trial.suggest_float("empty_sell_penalty", 1e-5, 1e-2)
    args.ent_coef = trial.suggest_float("ent_coef", 6e-5, 6e-2, log=True)
    args.gamma = trial.suggest_float("gamma", 0.98, 0.9999)
    args.hold_cost_rate = trial.suggest_float("hold_cost_rate", 1e-8, 1e-5, log=True)
    args.hold_incentive = trial.suggest_float("hold_incentive", 1e-10, 1e-7, log=True)
    args.illegal_buy_penalty = trial.suggest_float("illegal_buy_penalty", 1e-7, 1e-4)
    args.illegal_sell_penalty = trial.suggest_float("illegal_sell_penalty", 1e-4, 1e-1)
    args.learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-7, log=True)
    args.n_envs = trial.suggest_int("n_envs", 5, 10)
    args.n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    args.profit_bonus = trial.suggest_float("profit_bonus", 0.25, 1.0)
    args.window_size = trial.suggest_categorical("window_size", [30, 60, 120, 240])

    # Scale up timesteps for Standard Tuning
    args.n_rows = 160000  # 10000 / 20000  / 40000
    args.timesteps = 500000  # 30000 / 50000 / 100000
    args.dashboard = False

    try:
        # Pass the trial object directly into Python memory!
        score = run_experiment(args, trial=trial)
        return score
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 100.0  # baseline fallback


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
