#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --parquet-path "subset.parquet" --rows 30000 --timesteps 60000 --window-size 60 --data-seed 42 --dashboard True --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.000001 --empty-buy-penalty 0.005 --empty-sell-penalty 0.005 --illegal-sell-penalty 0.02 --budget-initial 100.0 --ent-coef 0.001 --gamma 0.95 --batch-size 512 --illegal-buy-penalty 0.02 --profit-bonus 0.15 --learning-rate 0.0006 --n-steps 2048 --clip-range 0.2 --action-space-type continuous --algorithm SAC
