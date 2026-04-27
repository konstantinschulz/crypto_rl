#!/usr/bin/env python3
import sys
print("optuna test start", file=sys.stderr)
try:
    import optuna
    print("optuna loaded successfully", file=sys.stderr)
    import subprocess
    import argparse
    import time
    print("modules loaded", file=sys.stderr)
except Exception as e:
    print(f"error loading optuna: {e}", file=sys.stderr)
    sys.exit(1)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    return lr * 2

if __name__ == "__main__":
    study = optuna.create_study()
    try:
        study.optimize(objective, n_trials=2)
        print(f"best value: {study.best_value}", file=sys.stderr)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)

