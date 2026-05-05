#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 5")
    print("Objective: Switch to standard PPO (MLP) for faster convergence, high reward scaling to force trades")
    print("====================================")

    cmd = [
        "/home/konstantin/dev/crypto_rl/.conda/bin/python",
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "ppo",                # Switched algorithm
        "--device", "cuda",
        "--days", "30",
        "--max-symbols", "5",
        "--train-steps", "50000",       # 50k is very fast with MLP
        "--learning-rate", "0.0003",
        "--batch-size", "64",
        "--n-steps", "512",
        "--n-epochs", "10",
        "--action-mapping-mode", "legacy",
        "--invalid-sell-mode", "force_buy",
        "--trade-action-bonus", "25.0",   # Huge bonus for trades
        "--inactivity-penalty", "-2.0",   # Huge penalty for doing nothing
        "--trade-execution-penalty", "0.0",
        "--reward-mode", "shaped",        # Force the environment to use the shaped rewards!
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

    print(f"Experiment 5 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

