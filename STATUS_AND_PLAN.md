# RL Trading Bot - Current Status & Next Steps

## 🎯 Objective
Train a reinforcement learning agent to profitably trade cryptocurrency futures using deep Q-learning and PPO algorithms.

## 📊 Current Status: **Debugging "No Trades" Issue**

### Problem
Model produces **0 trades** across all 3 backtest phases despite reward shaping improvements.

### Root Cause
- Model learns "do nothing" (hold) is locally optimal
- Starting portfolio ($100) at 0% loss is hard to beat
- Weak inactivity penalties (original: -0.1/step) negligible over 4000 steps
- Model can output "fake" actions (e.g., sell with 0% amount)

---

## 🔬 Experiments Completed

### Experiment 1: Improved Reward Shaping
- ✅ Lowered trading threshold: 0.1 → 0.05
- ✅ 5x stronger trade rewards: realized_pnl / 100.0 → realized_pnl / 20.0
- ❌ Result: **0 trades** (inactivity penalty too weak)

### Experiment 2: Aggressive Bonuses  
- ✅ Trade action bonus: 5.0
- ✅ Inactivity penalty: -0.1
- ❌ Result: **0 trades** (model still chose holding)

### Experiment 3: Ultra-Aggressive  
- ✅ Trade action bonus: 10.0
- ✅ Inactivity penalty: -2.0
- ⏳ Status: Attempted (terminal issues)

---

## ✨ Experiment 4 (Ready to Execute)

### Improvements
1. **Extreme Inactivity Penalty**: -5.0 per step
   - Total episode cost: -5.0 × 4000 steps = -20,000
   - Cost to avoid by trading once: Just 15.0 cost for +15.0 bonus = 0 net

2. **Invalid Action Penalties** (NEW)
   - Trying to sell with no positions: -3.0 penalty
   - Trying to sell invalid symbol: -2.0 penalty
   - Prevents model from exploiting "fake" actions

3. **Strong Trade Bonus**: +15.0 for any actual trade

### Configuration
```python
trade_action_bonus = 15.0
inactivity_penalty = -5.0
Invalid sell penalty = -3.0
```

### Expected Results
- Model learns to execute BUY and SELL actions
- Initial win rate: ~30-40% (random entry/exit)
- Total return: Negative (learning phase)
- Trades: **> 0** ✓

---

## 🚀 How to Run Experiment 4

### Step 1: Verify Environment
```bash
cd /home/konstantin/dev/crypto_rl
python test_env.py
```

Expected output:
```
✓ Successfully imported CryptoTradingEnv and TradingConfig
✓ Loaded data: XXXX rows, 3 symbols
✓ Created trading environment
✓ Environment reset successful
✓ Environment stepping works correctly
✓ All tests passed!
```

### Step 2: Run Experiment
```bash
python rl_trader.py \
  --days 14 \
  --max-symbols 3 \
  --train-steps 50000 \
  --mode backtest_3way \
  --learning-rate 0.0003 \
  --batch-size 16 \
  --n-steps 256 \
  --n-epochs 5
```

### Step 3: Check Results
```bash
# Monitor in real-time (if using --dashboard)
tail -f rl_dashboard_state.json

# Or check final summary
grep -A 5 "BACKTEST SUMMARY" exp4_results.log
```

### Expected Duration
- CPU: 3-5 minutes
- GPU: 1-2 minutes

---

## 📈 Next Phases (If Exp 4 Succeeds)

### Phase 2: Profitable Trading (Exp 5-6)
1. **Shift reward structure** from "trade at all costs" to "trade profitably"
2. **Add profit bonuses**: +50.0 for realized P&L > 0
3. **Add loss penalties**: -50.0 for realized P&L < 0
4. **Scale up data**: 14 → 21 days

### Phase 3: Production Ready (Exp 7-10)
1. **Larger models** ([256, 128, 64])
2. **More symbols** (5-10)
3. **Better hyperparameters** (tune learning rate, batch size)
4. **Risk limits** (max drawdown, position sizing)
5. **Backtesting** (1 year of data)

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `rl_trading_env.py` | RL environment with reward shaping |
| `rl_trader.py` | Training & evaluation pipeline |
| `test_env.py` | Quick validation script (NEW) |
| `EXPERIMENT_DIAGNOSTIC.md` | Root cause analysis (NEW) |
| `EXP4_PLAN.md` | Detailed Exp 4 plan (NEW) |
| `EXPERIMENT_LOG_v2.md` | Experiment results log |

---

## ✅ Checklist Before Running

- [ ] Files saved and no syntax errors
- [ ] `test_env.py` runs successfully
- [ ] Data file `binance_spot_1m_last4y_single.parquet` exists
- [ ] Conda environment activated
- [ ] ~30 minutes time available

---

## 📝 Success Criteria

### Minimum (Exp 4)
- ✓ Trades > 0
- ✓ No errors or crashes
- ✓ Training completes

### Target (Exp 5+)
- ✓ Trades: > 50 per 14-day period
- ✓ Win rate: > 50%
- ✓ Return: > 5% per 14-day period

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| 0 trades (again) | Increase inactivity_penalty to -10.0 |
| Memory error | Reduce --max-symbols or --days |
| Crashes | Run test_env.py to diagnose |
| Slow training | Use GPU with CUDA |

---

## 📚 References

- **Root Cause Analysis**: See `EXPERIMENT_DIAGNOSTIC.md`
- **Implementation Details**: See `rl_trading_env.py` lines 140-210
- **Full Backtest Protocol**: See `rl_trader.py` backtest_3way() method
- **Previous Experiments**: See `EXPERIMENT_LOG_v2.md`

---

**Last Updated**: March 30, 2026  
**Status**: Ready for Experiment 4  
**Next Action**: Run `python test_env.py` then `python rl_trader.py --days 14 --max-symbols 3 --train-steps 50000 --mode backtest_3way`

