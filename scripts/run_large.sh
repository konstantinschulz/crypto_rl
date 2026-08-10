#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --parquet-path "binance_spot_1m_last4y_single.parquet" --rows 40000 --timesteps 3000 --window-size 60 --data-seed 42 --dashboard True --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.000005 --empty-buy-penalty 0.080 --empty-sell-penalty 0.019 --budget-initial 100.0 --gamma 0.9944 --batch-size 256 --illegal-buy-penalty 0.009 --profit-bonus 0.37 --learning-rate 0.00002 --n-steps 1024 --clip-range 0.2 --action-space-type continuous --algorithm PPO --illegal-sell-penalty 0.045 --max-single-step-allocation 0.5 --action-dead-zone 0.15  # 400000 1200000
