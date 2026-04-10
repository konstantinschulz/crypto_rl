# RL Trading Bot - Experiment 4+ Plan

## Status
Currently debugging why the model produces 0 trades despite reward shaping improvements. Root cause identified: Model learns "do nothing" is locally optimal.

## Experiments Completed

| Exp | Approach | Result | Issue |
|-----|----------|--------|-------|
| 1 | Lower threshold + reward boost | 0 trades | Inactivity penalty too weak |
| 2 | +5.0 trade bonus, -0.1 inactivity | 0 trades | Still favors holding |
| 3 | +10.0 trade bonus, -2.0 inactivity | (attempted) | Needed stronger fix |
| 4 | **+15.0 bonus, -5.0 inactivity, invalid sell penalties** | **READY TO TEST** | TBD |

## Experiment 4 Details

### Configuration Changes
```python
TradingConfig:
  trade_action_bonus = 15.0      # Up from 5.0
  inactivity_penalty = -5.0      # Up from -0.1
  Invalid sell (no position) = -3.0 penalty  # NEW
  
TradingEnv.step():
  - Check if sell action is valid (has position in symbol)
  - If invalid: apply -3.0 penalty immediately
  - If valid: apply normal sell reward logic
```

### Mathematical Justification
- **Inactivity cost per episode**: -5.0 × 4000 steps = -20,000
- **Cost of 100 losing trades** (e.g., -0.01 each): -1.0
- **Ratio**: 20,000:1 strongly favors trading
- **Even bad trading is better than holding**

### Expected Outcome
1. Model learns to execute BUY actions (gets +15.0 bonus)
2. Model learns to execute SELL actions (gets +15.0 bonus)
3. Initial trades will likely be unprofitable (just learning mechanism)
4. Win rate expected: ~30-40% (random entry/exit)
5. Total return: Negative initially (learning phase)

### Success Criteria
✓ Trades > 0  
✓ Training continues without errors  
✓ Actions show [action_type > 0, symbol_idx, amount_pct > 0.05]

### Failure Modes
✗ Still 0 trades → Need curriculum learning or action masking
✗ Model collapses (neg infinity reward) → Penalties too high
✗ Memory errors → Data loading issue

---

## Experiment 5+ Pipeline (If Exp 4 Succeeds)

### Exp 5: Profitable Trading
**Goal**: Make the model actually make money, not just trade
```python
TradingConfig:
  trade_action_bonus = 5.0       # Reduce bonus (no longer need extreme)
  inactivity_penalty = -0.5      # Moderate penalty
  realized_pnl_bonus = 50.0      # STRONG bonus for profits
  realized_loss_penalty = -50.0  # STRONG penalty for losses (NEW)
```
**Rationale**: Shift from "trade at all costs" to "trade profitably"

### Exp 6: Scale Up
```python
days = 21                        # From 14 to 21
max_symbols = 5                  # From 3 to 5
train_steps = 100000             # From 50k to 100k
max_positions = 5                # From 3 to 5
```

### Exp 7: Model Architecture
```python
model_arch = [256, 128, 64]      # Larger network
learning_rate = 1e-4             # Lower LR for stability
n_steps = 512                    # Increase trajectory length
batch_size = 32                  # Increase batch size
```

---

## How to Run Experiment 4

```bash
cd /home/konstantin/dev/crypto_rl

# First verify environment works:
python test_env.py

# Then run experiment:
python rl_trader.py \
  --days 14 \
  --max-symbols 3 \
  --train-steps 50000 \
  --mode backtest_3way \
  --learning-rate 0.0003 \
  --batch-size 16 \
  --n-steps 256 \
  --n-epochs 5

# Monitor results:
tail -f exp4_results.log | grep -E "BACKTEST|Trades|Win Rate"
```

### Expected Runtime
- ~3-5 minutes on CPU  
- ~1-2 minutes on GPU

---

## Diagnostic Checklist

Before running Exp 4, verify:
- [ ] `rl_trading_env.py` loads without syntax errors
- [ ] TradingConfig has all fields defined
- [ ] _execute_buy() includes `self.total_trades += 1`
- [ ] Invalid action penalties in step() function
- [ ] No infinite loops or deadlocks in environment

---

## Files Modified

1. **rl_trading_env.py**
   - TradingConfig: inactivity_penalty = -5.0, trade_action_bonus = 15.0
   - step(): Added invalid action penalties and checks
   - _execute_buy(): Increments self.total_trades

2. **EXPERIMENT_LOG_v2.md** - Updated with findings

3. **EXPERIMENT_DIAGNOSTIC.md** - New analysis document

4. **test_env.py** - New validation script

---

## Key Insights Learned

1. **"Do Nothing" is Locally Optimal**: Starting with $100 and 0% loss is hard to beat
2. **Weak Penalties Fail**: -0.1 per step is negligible over 4000 steps
3. **Action Space Ambiguity**: Model can output "fake" actions (sell with 0 amount)
4. **Need Explicit Invalid Penalties**: Must penalize impossible actions

---

## Success Metrics (End Goal)

Once Exp 4 proves trading works, Exp 5+ should target:
- **Profitable**: Total Return > 5% per 14-day period
- **Consistent**: Win Rate > 50%
- **Efficient**: Sharpe Ratio > 1.0

Current Status: **0 trades → Must fix mechanism first**

Next Action: **Run Experiment 4 to verify trading mechanism works**

