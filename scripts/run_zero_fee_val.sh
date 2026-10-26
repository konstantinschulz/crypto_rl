#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Zero-fee diagnostic: does PPO find any gross edge over the equal-weight basket?
# Same data, folds and PPO settings as Optuna v4 trial 29 (val score −0.75 at
# 0.1% fees); only --fee-rate differs. The final report runs on the validation
# slices, so the test slice stays unread (ROADMAP P0-4), and the score is
# directly comparable to the trial's.
python "${REPO_ROOT}/main.py" --algorithm PPO --batch-size 128 --budget-initial 100.0 --checkpoint --clip-range 0.2 --cv-folds 3 --dashboard --data-seed 42 --decision-interval 15 --drawdown-penalty-coef 0.5 --ent-coef-final 0.0027662775419259023 --ent-coef-initial 0.0027662775419259023 --eval-freq auto --eval-length 20160 --fee-rate 0.0 --final-eval-split val --gamma 0.9925176264662898 --learning-rate 0.0009623030939973586 --max-asset-allocation 0.20 --max-checkpoints 5 --n-envs 9 --n-rows 18000000 --n-steps 1024 --net-arch-dim 256 --parquet-path "binance_spot_1m_last4y_single_htf.parquet" --rebalance-band 0.02 --reward-type excess_log_return --skip-multi-seed-eval --test-fraction 0.1 --timesteps 1200000 --train-chunks 5 --val-fraction 0.05 --weight-step 0.05 --window-size 240 "$@"
