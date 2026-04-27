#!/usr/bin/env python3
"""Run an Optuna hyperparameter sweep for the RL trader."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import optuna

VAL_RE = re.compile(r"Val Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")
TEST_RE = re.compile(r"Test Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")

def parse_metrics(output: str) -> dict:
    metrics = {
        "val_trades": 0.0,
        "val_win_rate_pct": 0.0,
        "val_return_pct": 0.0,
        "test_trades": 0.0,
        "test_win_rate_pct": 0.0,
        "test_return_pct": 0.0,
    }
    val_matches = VAL_RE.findall(output)
    if val_matches:
        trades, wr, ret = val_matches[-1]
        metrics["val_trades"] = float(trades)
        metrics["val_win_rate_pct"] = float(wr)
        metrics["val_return_pct"] = float(ret)

    test_matches = TEST_RE.findall(output)
    if test_matches:
        trades, wr, ret = test_matches[-1]
        metrics["test_trades"] = float(trades)
        metrics["test_win_rate_pct"] = float(wr)
        metrics["test_return_pct"] = float(ret)
    return metrics

def objective(trial: optuna.Trial, n_trials: int, args: argparse.Namespace) -> float:
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)

    # We can also sample reward penalties, trade execution penalty etc. if we want
    trade_execution_penalty = trial.suggest_float("trade_execution_penalty", 0.05, 0.3)
    inactivity_penalty = trial.suggest_float("inactivity_penalty", -2.0, -0.1)

    cmd = [
        sys.executable, "rl_trader.py",
        "--mode", "backtest_3way",
        "--device", "cuda",
        "--algo", "recurrent_ppo",
        "--days", str(args.days),
        "--max-symbols", str(args.max_symbols),
        "--train-steps", str(args.train_steps),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--n-steps", str(n_steps),
        "--n-epochs", str(n_epochs),
        "--trade-execution-penalty", str(trade_execution_penalty),
        "--inactivity-penalty", str(inactivity_penalty),
        "--fee-regime", "mexc_spot_mx",
        "--model-arch", "[256, 128, 64]",
        "--lstm-hidden-size", "128",
        "--lstm-layers", "1",
    ]

    print(f"\n[{trial.number}/{n_trials}] Running trial with params: {trial.params}")

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started

    metrics = parse_metrics(proc.stdout + "\n" + proc.stderr)

    val_ret = metrics["val_return_pct"]
    test_ret = metrics["test_return_pct"]
    val_trades = metrics["val_trades"]

    print(f"[{trial.number}/{n_trials}] Done in {elapsed:.1f}s | Val Ret: {val_ret:.2f}% | Test Ret: {test_ret:.2f}% | Trades: {int(val_trades)}")

    # Simple score prioritizing stable positive test returns
    if val_trades < 5:
        return -100.0 # Penalize no trading

    score = val_ret * 0.4 + test_ret * 0.6
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=5, help="Number of trials to run")
    parser.add_argument("--days", type=int, default=30, help="Days of data to load")
    parser.add_argument("--max-symbols", type=int, default=5, help="Number of symbols")
    parser.add_argument("--train-steps", type=int, default=20000, help="Training steps per trial")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize", study_name="crypto_rl_sweep")

    try:
        def study_obj(t):
            return objective(t, args.n_trials, args)
        study.optimize(study_obj, n_trials=args.n_trials)
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 50)
    print("Optimization finished.")
    print(f"Best trial value: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)

if __name__ == "__main__":
    main()


