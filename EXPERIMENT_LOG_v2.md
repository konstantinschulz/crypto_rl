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
