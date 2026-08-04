"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20 --rows 10000 --timesteps 30000
"""

from pathlib import Path
import optuna
import subprocess
import sys
import argparse

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
    args.profit_bonus = trial.suggest_float("profit_bonus", 0.0, 0.5)
    args.hold_cost_rate = trial.suggest_float("hold_cost_rate", 1e-7, 1e-5, log=True)
    args.empty_buy_penalty = trial.suggest_float("empty_buy_penalty", 1e-4, 1e-1)
    args.empty_sell_penalty = trial.suggest_float("empty_sell_penalty", 1e-4, 1e-1)
    args.illegal_buy_penalty = trial.suggest_float("illegal_buy_penalty", 1e-5, 1e-2)
    args.illegal_sell_penalty = trial.suggest_float("illegal_sell_penalty", 5e-4, 1e-1)
    args.action_dead_zone = trial.suggest_float("action_dead_zone", 0.05, 0.30)
    args.hold_incentive = trial.suggest_float("hold_incentive", 0.0, 0.002, log=True)
    args.lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    args.gamma = trial.suggest_float("gamma", 0.98, 0.999)
    args.n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    args.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    args.window_size = trial.suggest_categorical("window_size", [30, 60, 120, 240])
    # clip = trial.suggest_float("clip", 0.1, 0.4)

    # Scale up timesteps for Standard Tuning
    args.rows = 40000  # 10000 / 20000  / 40000
    args.timesteps = 150000  # 30000 / 50000 / 100000
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
    p.add_argument("--n-trials", type=int, default=40)
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
