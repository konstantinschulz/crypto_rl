# Experiment Diagnostic Report

**Date**: March 30, 2026  
**Issue**: Model consistently produces 0 trades despite reward shaping improvements

---

## Observations

### Exp 1: Improved Reward Shaping (0.05 threshold, stronger rewards)
- **Result**: 0 trades across all 3 phases (train, val, test)
- **Debug Output**: Raw actions show amount_pct=0
- **Training Reward**: Increased from 1.58e+04 to 1.65e+04 over training
- **Status**: ❌ FAILED - Reward shaping alone not sufficient

### Exp 2: Aggressive Bonuses (+5.0 for trade, -0.1 for inactivity)
- **Result**: 0 trades
- **Training Reward**: Increased to 4.06e+04
- **Debug Output**: action_type=2 (sell) but amount_pct=0
- **Status**: ❌ FAILED - Model chose "sell nothing" consistently

### Exp 3: Ultra-Aggressive (-2.0 inactivity penalty, +10.0 trade bonus)
- **Attempted but terminal issues encountered**

---

## Root Cause Analysis

The model learns optimal policy too quickly without enough exploration:
1. **Hold/Do Nothing is Locally Optimal**: 
   - Starting portfolio is already at $100
   - Holding has 0 risk
   - Trading has transaction costs (-0.1% fee)
   - Model learns holding is safer

2. **Action Space Saturation**:
   - action_type outputs near [0, 1, 2] bounds
   - When amount_pct=0, no actual trade executes
   - Model found a "free" action with no consequences

3. **Reward Signal Issues**:
   - Penalties not strong enough to overcome initial "do nothing" advantage
   - Realized PnL bonus only triggers on closed trades
   - Inactivity penalty (-2.0/step) should be stronger but environment isn't executing

---

## Root Cause: Action Validation Issue

Looking at the step() function:
```python
if action_type == 1 and amount_pct > 0.05:  # BUY
    ...
elif action_type == 2 and amount_pct > 0.05:  # SELL
    ...
else:
    # INACTIVITY PENALTY APPLIED HERE
    step_reward += self.config.inactivity_penalty
```

**Problem**: When action_type=2 (sell) but amount_pct=0:
- The condition `action_type == 2 and amount_pct > 0.05` is FALSE
- Code falls through to the else block
- Inactivity penalty IS applied
- But model still outputs action_type=2!

**This suggests**: The model is learning to output action_type=2 (sell) as a way to get the inactivity penalty, perhaps because:
1. There's some reward signal associated with the action output itself
2. The sell reward calculation is still giving some feedback

---

## Solution Approaches to Try

### Approach A: Explicit Action Type Penalty
Penalize "invalid" actions (e.g., selling when no positions exist):
```python
if action_type == 2 and len(self.positions) == 0:
    step_reward -= 5.0  # Heavy penalty for invalid action
```

### Approach B: Forced Exploration
Add noise to action selection during training to encourage trying buy/sell actions early:
```python
if np.random.random() < 0.2:  # 20% epsilon-greedy
    action_type = np.random.choice([0, 1, 2])
```

### Approach C: Curriculum Learning
Start training with artificially high prices for 1st symbol → model profitable → gradually remove subsidies

### Approach D: Alternative Reward Structure
Instead of inactivity penalty, use:
- Bonus for FIRST trade of episode (e.g., +20)
- Bonus for closing positions (e.g., +10)
- Penalty for max positions (no more buying)
- Only give returns reward when positions are held

### Approach E: Observation Space Enhancement
Include additional signals:
- "Buy opportunity score" (e.g., price momentum)
- "Volatility indicator"
- "Time since last trade"
These help model understand WHEN to trade, not just THAT it should trade

---

## Recommended Next Steps

1. **Implement Approach A**: Add penalty for invalid sell actions
2. **Test with smaller n_steps**: Reduce n_steps from 512 to 256 to force more frequent action queries
3. **Add debug logging**: Track percentage of invalid trades during training
4. **Test simple baseline**: Try algorithm that trades randomly vs. our learned policy
5. **Consider different action space**: Instead of [action_type, symbol, amount], try:
   - [buy_amount_pct, sell_positions_mask] - more explicit about what to do

---

## Files to Modify

- `rl_trading_env.py`: Add invalid action penalties, enhance observation space
- `rl_trader.py`: Optional - add training debug logging


