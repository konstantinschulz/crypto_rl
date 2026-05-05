#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 7")
    print("Objective: Switch to 'equity_delta' to make the agent prioritize ACTUAL portofolio growth while retaining legacy mapping.")
    print("====================================")

    cmd = [
        "/home/konstantin/dev/crypto_rl/.conda/bin/python",
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "ppo",
        "--device", "cuda",
        "--days", "30",
        "--max-symbols", "5",
        "--train-steps", "150000",       # Longer to learn PnL
        "--learning-rate", "0.0003",
        "--batch-size", "128",
        "--n-steps", "1024",
        "--n-epochs", "10",
        "--action-mapping-mode", "legacy",
        "--invalid-sell-mode", "force_buy",
        "--reward-mode", "equity_delta",    # True profit!
        "--fee-regime", "mexc_spot_mx",
        "--model-arch", "[256, 128]",       # Slightly larger model for true PnL
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with code {e.returncode}")
        sys.exit(1)

    print(f"Experiment 7 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

