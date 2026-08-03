#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --parquet-path "binance_spot_1m_last4y_single.parquet" --rows 60000 --timesteps 200000 --window-size 60 --data-seed 42 --dashboard True --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.000005 --empty-buy-penalty 0.023 --empty-sell-penalty 0.012 --budget-initial 100.0 --ent-coef 0.007 --gamma 0.9944 --batch-size 256 --illegal-buy-penalty 0.055 --profit-bonus 1.18 --learning-rate 0.00002 --n-steps 1024 --clip-range 0.2 --action-space-type continuous --algorithm SAC --illegal-sell-penalty 0.039 --max-single-step-allocation 0.5
