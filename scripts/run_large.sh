#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --action-dead-zone 0.10 --action-space-type continuous --algorithm PPO --batch-size 256 --budget-initial 100.0 --checkpoint --clip-range 0.11 --dashboard --empty-buy-penalty 0.072 --empty-sell-penalty 0.019 --fee-rate 0.001 --gamma 0.9942 --hold-cost-rate 0.000004 --illegal-buy-penalty 0.002 --illegal-sell-penalty 0.030 --learning-rate 0.00014 --max-checkpoints 5 --max-single-step-allocation 0.5 --n-envs 6 --n-steps 1024 --parquet-path "binance_spot_1m_last4y_single.parquet" --profit-bonus 0.41 --reward-type excess_return --rows 400000 --timesteps 1200000 --window-size 60 # 40000 / 400000 ; 3000 / 1200000 --data-seed 42 
