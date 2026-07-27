#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --parquet-path "subset.parquet" --rows 30000 --timesteps 60000 --window-size 60 --data-seed 42 --dashboard True --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.000004 --empty-buy-penalty 0.005 --empty-sell-penalty 0.005 --budget-initial 100.0 --ent-coef 0.007 --gamma 0.9944 --batch-size 256 --illegal-buy-penalty 0.02 --profit-bonus 1.29 --learning-rate 0.00002 --n-steps 1024 --clip-range 0.2 --action-space-type continuous --algorithm SAC --illegal-sell-penalty 0.02 --max-single-step-allocation 0.5
