"""
Automated hyper-parameter search using Optuna.
Usage:
    python scripts/optuna_search.py --n-trials 20 --rows 10000 --timesteps 30000
"""

import optuna
import subprocess
import sys
import argparse


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    ent = trial.suggest_float("ent", 1e-3, 0.05, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    clip = trial.suggest_float("clip", 0.1, 0.4)
    pb = trial.suggest_float("profit_bonus", 0.05, 0.5)

    cmd = [
        sys.executable,
        "main.py",
        "--rows",
        "10000",
        "--timesteps",
        "30000",
        "--learning-rate",
        str(lr),
        "--ent-coef",
        str(ent),
        "--gamma",
        str(gamma),
        "--n-steps",
        str(n_steps),
        "--clip-range",
        str(clip),
        "--profit-bonus",
        str(pb),
        "--dashboard",
        "False"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "Final Test Portfolio Value" in line:
            return float(line.split("$")[1])
    return 100.0  # no-trade baseline


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name="crypto_rl_v1",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
