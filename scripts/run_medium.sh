#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --parquet-path "subset.parquet" --rows 30000 --timesteps 60000 --window-size 60 --data-seed 42 --dashboard True --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.0000001 --empty-buy-penalty 0.005 --empty-sell-penalty 0.005 --illegal-sell-penalty 0.02 --budget-initial 100.0 --ent-coef 0.04 --gamma 0.9988 --batch-size 256 --illegal-buy-penalty 0.02 --profit-bonus 0.49 --learning-rate 0.002 --n-steps 1024 --clip-range 0.2 --action-space-type continuous --algorithm SAC
