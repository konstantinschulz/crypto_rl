#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 1")
    print("Objective: Test High Batch Size + Low Learning Rate + Targeted Penalties")
    print("====================================")

    cmd = [
        "/home/konstantin/dev/crypto_rl/.conda/bin/python",
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "recurrent_ppo",
        "--device", "cuda",
        "--days", "30",
        "--max-symbols", "5",
        "--train-steps", "30000",
        "--learning-rate", "0.0001",    # Slower learning
        "--batch-size", "128",          # More stable gradients
        "--n-steps", "1024",            # Longer trajectory
        "--n-epochs", "10",             # Optimize longer per batch
        "--trade-execution-penalty", "0.1",
        "--inactivity-penalty", "-0.8",
        "--fee-regime", "mexc_spot_mx",
        "--model-arch", "[256, 128, 64]",
        "--lstm-hidden-size", "128",
        "--lstm-layers", "1",
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with code {e.returncode}")
        sys.exit(1)

    print(f"Experiment 1 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

