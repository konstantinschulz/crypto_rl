#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 6")
    print("Objective: Fine-tune the penalties to reduce extreme trade churn and optimize for PnL")
    print("====================================")

    cmd = [
        "/home/konstantin/dev/crypto_rl/.conda/bin/python",
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "ppo",
        "--device", "cuda",
        "--days", "30",
        "--max-symbols", "5",
        "--train-steps", "100000",       # Train a bit longer natively
        "--learning-rate", "0.0003",
        "--batch-size", "64",
        "--n-steps", "512",
        "--n-epochs", "10",
        "--action-mapping-mode", "legacy",
        "--invalid-sell-mode", "force_buy",
        "--trade-action-bonus", "2.0",    # Reduced drastically from 25.0
        "--inactivity-penalty", "-0.5",   # Reduced from -2.0
        "--trade-execution-penalty", "0.05", # Re-introduced to discourage pure churn
        "--reward-mode", "shaped",
        "--fee-regime", "mexc_spot_mx",
        "--model-arch", "[128, 64]",
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with code {e.returncode}")
        sys.exit(1)

    print(f"Experiment 6 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

