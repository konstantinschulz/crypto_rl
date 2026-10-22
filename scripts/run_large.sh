#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/main.py" --algorithm PPO --batch-size 128 --budget-initial 100.0 --checkpoint --clip-range 0.2 --cv-folds 3 --dashboard --data-seed 42 --decision-interval 15 --ent-coef-final 0.01 --ent-coef-initial 0.01 --eval-freq auto --eval-length 10080 --fee-rate 0.001 --gamma 0.99 --learning-rate 0.0003 --max-asset-allocation 0.20 --max-checkpoints 5 --n-envs 9 --n-rows 18000000 --n-steps 1024 --net-arch-dim 256 --parquet-path "binance_spot_1m_last4y_single_htf.parquet" --rebalance-band 0.02 --reward-type excess_log_return --skip-multi-seed-eval --test-fraction 0.1 --timesteps 1200000 --train-chunks 1 --val-fraction 0.05 --weight-step 0.05 --window-size 240 # --n-rows 40000 / 400000 / 1200000 ; --timesteps 25000 / 1200000 --drawdown-penalty-coef 0.5 --skip-multi-seed-eval --eval-length 20160
# PPO settings are the ROADMAP P2-4 starting point; the old Optuna "best params" were
# tuned on training data with a leaky observation and are discarded.
