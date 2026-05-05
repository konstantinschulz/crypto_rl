--- manual_log.txt ---
MANUAL TUNING EXPERIMENT 1
Objective: Test High Batch Size + Low Learning Rate + Targeted Penalties
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo recurrent_ppo --device cuda --days 30 --max-symbols 5 --train-steps 30000 --learning-rate 0.0001 --batch-size 128 --n-steps 1024 --n-epochs 10 --trade-execution-penalty 0.1 --inactivity-penalty -0.8 --fee-regime mexc_spot_mx --model-arch [256, 128, 64] --lstm-hidden-size 128 --lstm-layers 1

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $100.00
  Max Portfolio Value: $100.00
  Total Reward: 0.00
  Steps: 576
  Trades: 0 | Win Rate: 0.0%
  Realized PnL: $0.00 | Unrealized: $0.00

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:    0 | WR:   0.0% | Return:   0.00%
Test Trades:   0 | WR:   0.0% | Return:   0.00%
======================================================================

Experiment 1 completed in 139.46 seconds.


--- manual_log_2.txt ---
MANUAL TUNING EXPERIMENT 2
Objective: Force trading by increasing inactivity penalty and reducing execution friction
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo recurrent_ppo --device cuda --days 30 --max-symbols 5 --train-steps 30000 --learning-rate 0.0001 --batch-size 128 --n-steps 1024 --n-epochs 10 --trade-execution-penalty 0.0 --inactivity-penalty -5.0 --trade-action-bonus 25.0 --fee-regime mexc_spot_mx --model-arch [256, 128, 64] --lstm-hidden-size 128 --lstm-layers 1

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $100.00
  Max Portfolio Value: $100.00
  Total Reward: 0.00
  Steps: 576
  Trades: 0 | Win Rate: 0.0%
  Realized PnL: $0.00 | Unrealized: $0.00

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:    0 | WR:   0.0% | Return:   0.00%
Test Trades:   0 | WR:   0.0% | Return:   0.00%
======================================================================

Experiment 2 completed in 144.30 seconds.


--- manual_log_3.txt ---
MANUAL TUNING EXPERIMENT 3
Objective: Adopt baseline structural params from Exp10 but focus on tuning batch_size + LR
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo recurrent_ppo --device cuda --days 30 --max-symbols 5 --train-steps 30000 --learning-rate 0.0003 --batch-size 32 --n-steps 512 --n-epochs 5 --action-mapping-mode legacy --invalid-sell-mode force_buy --trade-action-bonus 8.5 --inactivity-penalty -0.9 --trade-execution-penalty 0.11 --trade-rate-penalty 0.22 --target-trade-rate 0.11 --trade-rate-window 240 --reward-equity-delta-scale 100.0 --turnover-penalty-rate 0.14 --continuous-drawdown-penalty 5.0 --min-hold-steps 4 --trade-cooldown-steps 1 --max-trades-per-window 96 --trade-window-steps 240 --fee-regime mexc_spot_mx --model-arch [256, 128, 64] --lstm-hidden-size 128 --lstm-layers 1

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $100.00
  Max Portfolio Value: $100.00
  Total Reward: 0.00
  Steps: 576
  Trades: 0 | Win Rate: 0.0%
  Realized PnL: $0.00 | Unrealized: $0.00

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:    0 | WR:   0.0% | Return:   0.00%
Test Trades:   0 | WR:   0.0% | Return:   0.00%
======================================================================

Experiment 3 completed in 146.74 seconds.


--- manual_log_4.txt ---
MANUAL TUNING EXPERIMENT 4
Objective: Increase Train Steps and reduce Batch Size for vastly more gradient updates.
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo recurrent_ppo --device cuda --days 30 --max-symbols 5 --train-steps 100000 --learning-rate 0.0003 --batch-size 16 --n-steps 256 --n-epochs 5 --action-mapping-mode legacy --invalid-sell-mode force_buy --trade-action-bonus 10.0 --inactivity-penalty -1.0 --trade-execution-penalty 0.1 --fee-regime mexc_spot_mx --model-arch [256, 128, 64] --lstm-hidden-size 128 --lstm-layers 1

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $100.00
  Max Portfolio Value: $100.00
  Total Reward: 0.00
  Steps: 576
  Trades: 0 | Win Rate: 0.0%
  Realized PnL: $0.00 | Unrealized: $0.00

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:    0 | WR:   0.0% | Return:   0.00%
Test Trades:   0 | WR:   0.0% | Return:   0.00%
======================================================================

Experiment 4 completed in 457.75 seconds.


--- manual_log_5.txt ---
MANUAL TUNING EXPERIMENT 5
Objective: Switch to standard PPO (MLP) for faster convergence, high reward scaling to force trades
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo ppo --device cuda --days 30 --max-symbols 5 --train-steps 50000 --learning-rate 0.0003 --batch-size 64 --n-steps 512 --n-epochs 10 --action-mapping-mode legacy --invalid-sell-mode force_buy --trade-action-bonus 25.0 --inactivity-penalty -2.0 --trade-execution-penalty 0.0 --reward-mode shaped --fee-regime mexc_spot_mx --model-arch [128, 64]

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $99.75
  Max Portfolio Value: $100.08
  Total Reward: 14080.51
  Steps: 576
  Trades: 580 | Win Rate: 56.0%
  Realized PnL: $-0.22 | Unrealized: $-0.03

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:  577 | WR:  25.6% | Return:  -0.01%
Test Trades: 580 | WR:  56.0% | Return:  -0.25%
======================================================================

Experiment 5 completed in 118.29 seconds.


--- manual_log_6.txt ---
MANUAL TUNING EXPERIMENT 6
Objective: Fine-tune the penalties to reduce extreme trade churn and optimize for PnL
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo ppo --device cuda --days 30 --max-symbols 5 --train-steps 100000 --learning-rate 0.0003 --batch-size 64 --n-steps 512 --n-epochs 10 --action-mapping-mode legacy --invalid-sell-mode force_buy --trade-action-bonus 2.0 --inactivity-penalty -0.5 --trade-execution-penalty 0.05 --reward-mode shaped --fee-regime mexc_spot_mx --model-arch [128, 64]

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $99.75
  Max Portfolio Value: $100.08
  Total Reward: 803.71
  Steps: 576
  Trades: 580 | Win Rate: 56.0%
  Realized PnL: $-0.22 | Unrealized: $-0.03

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:  577 | WR:  25.6% | Return:  -0.01%
Test Trades: 580 | WR:  56.0% | Return:  -0.25%
======================================================================

Experiment 6 completed in 97.26 seconds.


--- manual_log_7.txt ---
MANUAL TUNING EXPERIMENT 7
Objective: Switch to 'equity_delta' to make the agent prioritize ACTUAL portofolio growth while retaining legacy mapping.
====================================
Running command: /home/konstantin/dev/crypto_rl/.conda/bin/python rl_trader.py --mode backtest_3way --algo ppo --device cuda --days 30 --max-symbols 5 --train-steps 150000 --learning-rate 0.0003 --batch-size 128 --n-steps 1024 --n-epochs 10 --action-mapping-mode legacy --invalid-sell-mode force_buy --reward-mode equity_delta --fee-regime mexc_spot_mx --model-arch [256, 128]

Loading 30 days of data...
✓ Loaded 14405 rows from 5 symbols

Fee regime: mexc_spot_mx | buy_fee=0.040% | sell_fee=0.040%

...
  Final Portfolio Value: $98.94
  Max Portfolio Value: $100.41
  Total Reward: -200.00
  Steps: 576
  Trades: 4 | Win Rate: 0.0%
  Realized PnL: $-1.06 | Unrealized: $-0.01

======================================================================
BACKTEST SUMMARY
======================================================================
Val Trades:    4 | WR:   0.0% | Return:  -0.88%
Test Trades:   4 | WR:   0.0% | Return:  -1.06%
======================================================================

Experiment 7 completed in 110.19 seconds.


