#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --rows 30000 --timesteps 60000 --window-size 60 --data-seed 42 --dashboard --reward-type excess_return --fee-rate 0.001 --hold-cost-rate 0.0002 --empty-buy-penalty 0.005 --empty-sell-penalty 0.005 --illegal-sell-penalty 0.02 --budget-initial 100.0 --ent-coef 0.025 --gamma 0.95 --batch-size 512 --illegal-buy-penalty 0.02 --trade-freq-incentive 0.01  --profit-bonus 0.15
