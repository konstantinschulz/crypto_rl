#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 4")
    print("Objective: Increase Train Steps and reduce Batch Size for vastly more gradient updates.")
    print("====================================")

    cmd = [
        "/home/konstantin/dev/crypto_rl/.conda/bin/python",
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "recurrent_ppo",
        "--device", "cuda",
        "--days", "30",
        "--max-symbols", "5",
        "--train-steps", "100000",   # Tripled the train steps
        "--learning-rate", "0.0003",
        "--batch-size", "16",        # Reduced batch size to get more updates
        "--n-steps", "256",          # Shorter horizons
        "--n-epochs", "5",
        "--action-mapping-mode", "legacy",
        "--invalid-sell-mode", "force_buy",
        "--trade-action-bonus", "10.0",
        "--inactivity-penalty", "-1.0",
        "--trade-execution-penalty", "0.1",
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

    print(f"Experiment 4 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

