# Model Optimization Experiments - Results Log v2.1

**Date**: March 30, 2026  
**Objective**: Improve model profitability from non-profitable baseline (14-day, 3-symbol) model

---

## Problem Identified

### The "No Trades" Issue
Despite multiple reward shaping improvements, the model consistently produces 0 trades:
- **Exp 1** (Improved thresholds): 0 trades
- **Exp 2** (Aggressive bonuses): 0 trades  
- **Exp 3** (Ultra-aggressive): 0 trades (attempted)

### Root Cause
Model learns to output action_type=2 (sell) with amount_pct=0:
- This is an "invalid" action (selling when no positions exist)
- Falls through to inactivity penalty (but still gets some reward signal)
- **Why**: Inactivity penalty not strong enough vs. transaction cost avoidance

---

## Solutions Implemented (Exp 4+)

### Fix 1: Invalid Action Penalties
```python
# Added penalty for trying to sell without positions
if action_type == 2 and len(self.positions) == 0:
    step_reward -= 2.0  # Extra penalty
    
# Added penalty for invalid sell (no position in symbol)
if sell_reward == 0:  # No actual position sold
    step_reward -= 3.0
```

### Fix 2: Extreme Inactivity Penalty
- **Old**: inactivity_penalty = -0.1 (too weak)
- **New**: inactivity_penalty = -5.0 per step (extreme - costs ~-20,000 per ~4k-step episode)
- **Trade Bonus**: Increased from 5.0 to 15.0 per trade

### Rationale
- At -5.0/step × 4000 steps/episode = -20,000 total penalty
- Even if all trades lose -0.01 each, trading 100 times = -1.0 loss
- Holding entire episode = -20,000 loss
- **Math favors trading heavily**

---

## Experiment 4: Extreme Penalties + Invalid Action Penalties

**Configuration**:
```
trade_action_bonus = 15.0      (was 5.0 in Exp 1)
realized_pnl_bonus = 100.0
inactivity_penalty = -5.0      (was -0.1 in Exp 1)
Invalid sell penalty = -3.0    (NEW)
```

**Expected Results**:
- Model MUST trade to avoid -5.0 penalty every step
- Invalid sells heavily penalized (-3.0)
- Valid buys heavily rewarded (+15.0)

**Hypothesis**: 
- Model will learn to execute buys and sells  
- Will likely be unprofitable at first (learning to trade, not trading well)
- But proves the mechanism works

---

## Experiment 4 Result (Actual)

**Run command**:
```bash
python -u rl_trader.py --days 14 --max-symbols 3 --train-steps 50000 --mode backtest_3way
```

**Observed outcome**:
- Val Trades: **0** | WR: **0.0%** | Return: **0.00%**
- Test Trades: **0** | WR: **0.0%** | Return: **0.00%**
- Eval mean reward at 50k: **~ -12093** (penalties dominate)
- Policy still outputs mostly `action_type=2` with `amount_pct=0`

**Conclusion**: Exp 4 did **not** solve trade execution. Penalty-only shaping is insufficient with current action parameterization.

## Updated Next Steps (Exp 5)

1. Implement **action masking / validity gating** in `step()`:
   - Block `sell` when no open positions
   - If invalid action selected, remap to `hold` (or forced `buy`) before reward calc
2. Split action into discrete-valid pathways (recommended):
   - Separate discrete `action_type` from constrained amount behavior
   - Enforce per-action valid domains before execution
3. Add diagnostics to training/eval logs:
   - `% invalid_sell_attempts`
   - `% buy_actions`, `% sell_actions`, `% hold_actions`
   - `% actions remapped by validator`
4. Re-run with a short gate test (10k steps), then full 50k if invalid-action rate drops.

## Additional Note

Even with heavy inactivity penalties, a continuous-box action head can still converge to degenerate boundary behavior. Next iteration should prioritize **valid action enforcement** over stronger scalar penalties.

---

## Files Modified

- `rl_trading_env.py`:
  - Added invalid action penalties
  - Increased trade_action_bonus to 15.0
  - Increased inactivity_penalty to -5.0
  - Enhanced step() logic to check for invalid sells

---

## Experiment 5 Gate Test (10k) - Validity Remap to Forced Buy

**Code change**:
- Invalid `sell` with no positions is remapped to `buy` (instead of `hold`)

```python
if action_type == 2 and len(self.positions) == 0:
    action_type = 1
```

**Run command**:
```bash
python -u rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way
```

**Observed outcome**:
- Val Trades: **4032** | WR: **39.8%** | Return: **-0.61%**
- Test Trades: **4034** | WR: **34.7%** | Return: **-0.65%**
- Trade mechanism restored: model no longer stuck at zero-trade behavior
- Side effect: extreme overtrading (nearly one trade per step)

**Conclusion**:
- ✅ Action validity remap solved the no-trade collapse.
- ⚠️ Policy now overtrades and remains slightly unprofitable due to fees/churn.

## Experiment 5 Full Run (50k) - Confirmation

**Run command**:
```bash
python -u rl_trader.py --days 14 --max-symbols 3 --train-steps 50000 --mode backtest_3way
```

**Observed outcome**:
- Val Trades: **4032** | WR: **39.8%** | Return: **-0.61%**
- Test Trades: **4034** | WR: **34.7%** | Return: **-0.65%**
- Metrics are effectively identical to the 10k gate test, confirming a stable overtrading policy.

**Interpretation**:
- Forced-buy remap fixes action inactivity but dominates policy behavior.
- The next iteration must reduce action-forcing incentives and reweight toward quality (profit per trade), not quantity.

## Next Tuning (Exp 6)

1. Reduce churn incentives:
   - Lower `trade_action_bonus` (e.g., 15.0 -> 1.0 to 3.0)
   - Reduce/remodel inactivity penalty (e.g., -5.0 -> -0.05 to -0.2)
2. Add minimum holding horizon before sell reward bonus to discourage micro-flips.
3. Add trade frequency penalty per step when recent trades exceed threshold.
4. Re-run 10k gate, then 50k full run after trade-rate drops below ~10-20% of steps.

---

## Notes

The core insight: **Model learns locally optimal "do nothing" strategy**
- Starting with $100 and no losses is hard to beat
- Removing inactivity penalty entirely makes holding cost too much
- Adding explicit invalid-action penalties prevents "fake" trading

---

## Experiment 6 Campaign (April 2026) - Profitability-Oriented Tuning

### Goal
Move from the stable-but-unprofitable overtrading policy (about -0.65% on test) toward non-negative/positive test return.

### Code/Config Enablers Added
To run controlled experiments without editing code each time, added CLI knobs in `rl_trader.py` + `TradingConfig` fields in `rl_trading_env.py`:

- `--trade-action-bonus`
- `--inactivity-penalty`
- `--invalid-sell-mode` (`force_buy|hold|penalize`)
- `--invalid-sell-penalty`
- `--trade-execution-penalty`
- `--max-budget-per-trade`
- `--transaction-cost`

### Fresh Baseline (control)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu
```

Result:
- Val: 4032 trades | WR 39.8% | Return **-0.61%**
- Test: 4034 trades | WR 34.7% | Return **-0.65%**

### Exp 6A - Hold remap + low-churn rewards
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode hold --trade-action-bonus 2.0 --inactivity-penalty -0.2 \
  --invalid-sell-penalty 1.0 --trade-execution-penalty 0.05 --learning-rate 0.0002 \
  --n-epochs 4 --batch-size 32
```

Result:
- Val/Test trades: **0**
- Return: **0.00%**

Conclusion:
- Failure mode returned to no-trade collapse.

### Exp 6B - Keep force-buy, add churn penalties
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode force_buy --trade-action-bonus 1.5 --inactivity-penalty -0.5 \
  --invalid-sell-penalty 0.5 --trade-execution-penalty 0.2 --learning-rate 0.0002 \
  --n-epochs 4 --batch-size 32
```

Result:
- Val/Test almost unchanged from baseline
- Test return remained near **-0.65%**

Conclusion:
- Failure: force-buy remap still dominates behavior and keeps extreme trade count.

### Exp 6C - Penalize invalid sells (no remap)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode penalize --trade-action-bonus 4.0 --inactivity-penalty -0.3 \
  --invalid-sell-penalty 6.0 --trade-execution-penalty 0.02 --learning-rate 0.00025 \
  --n-epochs 5 --batch-size 32
```

Result:
- Val/Test trades: **0**
- Return: **0.00%**

Conclusion:
- Failure: policy avoids trading under penalize-only invalid handling.

### Exp 6D - Strong incentives + hold remap
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode hold --trade-action-bonus 8.0 --inactivity-penalty -2.0 \
  --invalid-sell-penalty 4.0 --trade-execution-penalty 0.1
```

Result:
- Val/Test trades: **0**
- Return: **0.00%**

Conclusion:
- Failure: hold remap consistently risks no-trade convergence in this action setup.

### Exp 6F - Lower costs + smaller per-trade budget
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode force_buy --trade-action-bonus 15.0 --inactivity-penalty -5.0 \
  --max-budget-per-trade 10 --transaction-cost 0.0002
```

Result:
- Val: 4032 trades | WR 52.1% | Return **-0.11%**
- Test: 4034 trades | WR 41.6% | Return **-0.16%**

Conclusion:
- Significant improvement vs baseline, still slightly negative.

### Exp 6G (Best So Far) - Very low cost + smaller budget
Gate (10k):
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --invalid-sell-mode force_buy --trade-action-bonus 15.0 --inactivity-penalty -5.0 \
  --max-budget-per-trade 8 --transaction-cost 0.00005
```

Full confirmation (50k):
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 50000 --mode backtest_3way --device cpu \
  --invalid-sell-mode force_buy --trade-action-bonus 15.0 --inactivity-penalty -5.0 \
  --max-budget-per-trade 8 --transaction-cost 0.00005
```

Result (stable across 10k/50k):
- Val: 4032 trades | WR 53.1% | Return **-0.02%**
- Test: 4033 trades | WR 46.4% | Return **+0.03%**

Conclusion:
- First tested configuration with **positive test return**.
- Caveat: profitability is achieved under a **much lower transaction-cost assumption** than baseline.

### Key Takeaways
1. Invalid-sell handling is still the dominant behavioral control:
   - `force_buy` => reliable trading but overtrading.
   - `hold`/`penalize` => frequent no-trade collapse.
2. Within current action-space design, reward-only tuning did not remove overtrading robustly.
3. Best measured profitability came from reducing trading friction (`transaction_cost`) and risk per trade (`max_budget_per_trade`).
4. Next major step should be structural:
   - redesign action parameterization (discrete valid action head),
   - or add explicit trade-frequency regularization tied to recent activity.

---

## Experiment 7 Campaign (April 2026) - MEXC-Realistic Fees + Structural Controls

### Requested goals
1. Use a more realistic MEXC fee regime.
2. Implement structural action mapping to reduce invalid-action pathologies.
3. Add explicit trade-frequency regularization.
4. Run robustness sweep on more data/symbols.

### Implementation changes

#### A) Realistic fee regime support (MEXC)
- Added `fee_regime`, `buy_fee_rate`, `sell_fee_rate` to `TradingConfig`.
- Added CLI in `rl_trader.py`:
  - `--fee-regime {mexc_spot_standard,mexc_spot_mx,legacy}`
  - `--transaction-cost` (manual override)
  - `--buy-fee-rate`, `--sell-fee-rate` (manual override)
- Default now targets MEXC spot-standard style fees (`0.05%` buy / `0.05%` sell) unless overridden.

#### B) Structural action mapping
- Added `action_mapping_mode` in `TradingConfig`:
  - `legacy`: old `[action_type, symbol_idx, amount]`
  - `validity_constrained`: uses intent in `[-1,1]` with deadzone and valid-state-aware sell mapping
- Added controls:
  - `--action-mapping-mode`
  - `--action-deadzone`

#### C) Trade-frequency regularization
- Added rolling trade-rate tracking (`deque`) in env.
- Added penalty when recent trade-rate exceeds target:
  - `--trade-rate-window`
  - `--target-trade-rate`
  - `--trade-rate-penalty`

---

### Exp 7A - MEXC fees + validity-constrained action mapping (structural baseline)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --dashboard --dashboard-state rl_dashboard_state_exp7a.json \
  --fee-regime mexc_spot_standard \
  --action-mapping-mode validity_constrained \
  --trade-rate-penalty 0.0
```

Result:
- Val/Test trades: **0**
- Return: **0.00%**

Conclusion:
- Failure: the constrained mapping (as currently parameterized) collapses to no-trade policy.

### Exp 7B - MEXC fees + legacy mapping + trade-rate regularization
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --dashboard --dashboard-state rl_dashboard_state_exp7b.json \
  --fee-regime mexc_spot_standard \
  --action-mapping-mode legacy \
  --trade-rate-penalty 2.0 --target-trade-rate 0.12 --trade-rate-window 240
```

Result:
- Val: 4032 trades | WR 46.95% | Return **-0.30%**
- Test: 4034 trades | WR 38.45% | Return **-0.34%**

Conclusion:
- Fees realistic to MEXC improved loss vs the old 0.1% assumption, but still unprofitable.
- Regularizer did not reduce trade count materially (still near one trade per step).

### Exp 7E (control) - same as 7B but no trade-rate penalty
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --dashboard --dashboard-state rl_dashboard_state_exp7e.json \
  --fee-regime mexc_spot_standard \
  --action-mapping-mode legacy \
  --trade-rate-penalty 0.0
```

Result:
- Test return also **-0.34%** (essentially unchanged vs Exp 7B)

Conclusion:
- The rate regularizer failed. The legacy mapping combined with strong `force_buy` dominates everything.

---

## High-Volume Manual Tuning Exp 1-7 (April 2026)

To rigorously isolate exactly how hyperparameters, algorithms, and rewards interact, we launched targeted scripts (`run_manual_tuning_expX.py`).

### Exps 1 to 4: Recurrent PPO Collapse
We tried utilizing `recurrent_ppo` (LSTM) alongside high/low batch sizes, training steps ranging from 30,000 to 100,000, and massive inactivity penalties.
**Results**: Every run ended in **0 Trades** on the Val and Test sets, achieving a 0.00% return.
**Interpretation**: The LSTM policies were extremely unstable or prone to avoiding all potential loss vectors immediately. Even massive `inactivity_penalty=-5.0` parameters forced extreme negative episode returns during train mode without actually producing trading edges that translated to the evaluation phases.

### Exps 5 to 6: Switching to Standard MLP PPO
Because `recurrent_ppo` was failing completely on evaluation translation:
We switched to `--algo ppo` representing heavily scaled MLP layers `[128, 64]`. Combined with heavy reward shaping (`--trade-action-bonus 25` down to `2`) and `invalid-sell-mode force_buy`.
**Results**: The standard PPO networks immediately unlocked trading behavior. We saw ~580 trades (almost exactly 1 per step on the eval sequence), achieving a 56% Win Rate. However, Test Return persisted precisely around **-0.25%**. 
**Interpretation**: The agent found the structurally optimal loophole: rapidly overtrading to capture artificial step bonuses, but losing just enough real money to MEXC proxy fees that the final portfolio delta went negative.

### Exp 7: Pure Equity Delta & Reality Check
To shatter the exploit, we stripped out the artificial `"shaped"` reward completely. We set `--reward-mode equity_delta`, telling the PPO (trained for 150k steps) that only pure, actual portfolio balance changes matter.
**Result**: 
- Val Trades Drops: **4** | Return: **-0.88%**
- Test Trades Drops: **4** | Return: **-1.06%**
**Interpretation**: The overtrading completely disappeared. The PPO model successfully deduced that high churn directly depletes PnL through fees. However, its small handful of trades were losers. The model is managing risk but hasn't discovered a profitable predictive edge.

---

## Strategic Summary and Concrete Next Steps

**The Current Paradigm is Exhausted**: 
We have definitively proved that standard `PPO` can map to sensible financial metrics when constrained strictly by `equity_delta`. However, small runs capping at 100k-150k steps over 14-30 day horizons will never yield fundamental alphas in highly efficient market proxies. It has simply learned to not pay fees.

### Recommended Action Plan

We must execute the Phase 1 "Massive Scale-up" directives documented in `FUTURE_DIRECTIONS.md`. 
1. **Increase Total Timesteps**: Ramp from 150k steps to **2-5 Million** steps. RL requires massive trajectory diversity to discover robust alpha vectors (predicting price rather than just penalizing action bounds).
2. **Expand Context Duration**: Broaden `--days` to 100-365 days and scale up the max symbols (10-20+). Training on 30 days forces catastrophic overfitting to singular micro trends.
3. **Parallel Subprocess Vector Environments**: Implement `--num-envs 8` using `SubprocVecEnv` so 8 parallel instances collect broad variance simultaneously, stabilizing gradients over the huge step counts.
4. **Target Script**: Scaffold a `run_heavy_training.py` that utilizes pure MLP PPO, `equity_delta`, and parallel workers for overnight tracking.
