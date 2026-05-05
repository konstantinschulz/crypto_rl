#!/usr/bin/env python3
"""Run a targeted manual hyperparameter tuning experiment."""

import subprocess
import sys
import time

def main():
    print("====================================")
    print("MANUAL TUNING EXPERIMENT 3")
    print("Objective: Adopt baseline structural params from Exp10 but focus on tuning batch_size + LR")
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
        "--learning-rate", "0.0003",    # Slightly more aggressive than before
        "--batch-size", "32",           # Standard batch size
        "--n-steps", "512",
        "--n-epochs", "5",
        # Structural params from known good runs:
        "--action-mapping-mode", "legacy",
        "--invalid-sell-mode", "force_buy",
        "--trade-action-bonus", "8.5",
        "--inactivity-penalty", "-0.9",
        "--trade-execution-penalty", "0.11",
        "--trade-rate-penalty", "0.22",
        "--target-trade-rate", "0.11",
        "--trade-rate-window", "240",
        "--reward-equity-delta-scale", "100.0",
        "--turnover-penalty-rate", "0.14",
        "--continuous-drawdown-penalty", "5.0",
        "--min-hold-steps", "4",
        "--trade-cooldown-steps", "1",
        "--max-trades-per-window", "96",
        "--trade-window-steps", "240",
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

    print(f"Experiment 3 completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()

