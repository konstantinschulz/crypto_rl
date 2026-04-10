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
- Failure: current trade-rate penalty formulation is too weak against action-bonus incentives.

### Exp 7D - validity-constrained stress test (small deadzone, stronger action bonus)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --dashboard --dashboard-state rl_dashboard_state_exp7d.json \
  --fee-regime mexc_spot_standard \
  --action-mapping-mode validity_constrained --action-deadzone 0.02 \
  --trade-action-bonus 20 --inactivity-penalty -4 \
  --trade-rate-penalty 0.5 --target-trade-rate 0.2
```

Result:
- Val/Test trades: **0**
- Return: **0.00%**

Conclusion:
- Failure: no-trade collapse remains for constrained mapping in this form.

### Exp 7C - robustness sweep (more days/symbols)
Command:
```bash
python rl_trader.py --days 30 --max-symbols 5 --train-steps 10000 --mode backtest_3way --device cpu \
  --dashboard --dashboard-state rl_dashboard_state_exp7c.json \
  --fee-regime mexc_spot_standard \
  --action-mapping-mode legacy \
  --trade-rate-penalty 2.0 --target-trade-rate 0.12 --trade-rate-window 240
```

Result:
- Val: 8640 trades | WR 39.26% | Return **-0.55%**
- Test: 8641 trades | WR 38.73% | Return **-0.88%**

Conclusion:
- Robustness failure: performance degrades when scaling data breadth.

---

### Current status after Exp 7
1. **MEXC-realistic fees are now implemented and defaultable**.
2. **Step 1 (structural action mapping) implemented**, but current variant tends to no-trade collapse.
3. **Step 2 (trade-rate regularization) implemented**, but not yet strong enough to reduce overtrading under legacy mapping.
4. **Step 3 (robustness sweep) completed**, and showed worse returns on broader data.

### Recommended follow-up
- Replace reward shaping with stronger policy-structure constraints:
  - separate valid discrete action head from sizing head,
  - or enforce hard min-hold + cooldown constraints at environment level,
  - and tune trade-rate penalty jointly with lower trade-action bonus.

---

## Experiment 8 Campaign (April 2026) - Hard Constraints + MEXC-Realistic Fees

### Goal
Push profitability further by combining:
1. realistic MEXC fees,
2. hard environment-level anti-churn constraints,
3. reward/regularization tuning,
4. robustness validation.

### New code implemented (Exp 8)

#### Environment hard constraints (`rl_trading_env.py`)
Added new `TradingConfig` fields and enforcement logic:
- `min_hold_steps`
- `trade_cooldown_steps`
- `max_trades_per_window`
- `trade_window_steps`
- `constraint_violation_penalty`

Implementation notes:
- Added rolling executed-trade window via `deque`.
- Added `_can_execute_trade(...)` gate in `step()`.
- If constraints fail, action is remapped to hold and optional penalty is applied.

#### CLI support (`rl_trader.py`)
Added pass-through args:
- `--min-hold-steps`
- `--trade-cooldown-steps`
- `--max-trades-per-window`
- `--trade-window-steps`
- `--constraint-violation-penalty`

Also changed default mapping to:
- `--action-mapping-mode legacy` (because `validity_constrained` frequently collapsed to no-trade in Exp 7).

---

### Exp 8A - Baseline under realistic MEXC spot standard fees
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --fee-regime mexc_spot_standard --action-mapping-mode legacy \
  --min-hold-steps 0 --trade-cooldown-steps 0 --max-trades-per-window 0
```

Result:
- Test return: **-0.3416%**
- Test trades: **4034**

### Exp 8B - Moderate hard constraints + trade-rate regularization
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --fee-regime mexc_spot_standard --action-mapping-mode legacy \
  --trade-rate-penalty 0.8 --target-trade-rate 0.12 --trade-rate-window 240 \
  --min-hold-steps 10 --trade-cooldown-steps 3 --max-trades-per-window 48 --trade-window-steps 240 \
  --constraint-violation-penalty 0.3
```

Result:
- Test return: **-1.1887%**
- Test trades: **10**

Conclusion:
- Over-constrained: severe undertrading and worse returns.

### Exp 8C - Strong hard constraints
Command (stronger than 8B):
```bash
python rl_trader.py ... --min-hold-steps 30 --trade-cooldown-steps 8 --max-trades-per-window 24 \
  --trade-rate-penalty 1.5 --target-trade-rate 0.08 --constraint-violation-penalty 0.6
```

Result:
- Test return: **-1.4930%**
- Test trades: **10**

Conclusion:
- Failure: hard caps too aggressive.

### Exp 8D - Constraint + reward reweighting (quality-first)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --fee-regime mexc_spot_standard --action-mapping-mode legacy \
  --trade-action-bonus 6 --inactivity-penalty -1.5 --trade-execution-penalty 0.25 \
  --trade-rate-penalty 1.0 --target-trade-rate 0.10 \
  --min-hold-steps 20 --trade-cooldown-steps 5 --max-trades-per-window 32 \
  --constraint-violation-penalty 0.5
```

Result:
- Test return: **0.00%**
- Test trades: **0**

Conclusion:
- Failure mode switched to no-trade collapse.

### Exp 8E - Mild constraints (best under MEXC standard)
Command:
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --fee-regime mexc_spot_standard --action-mapping-mode legacy \
  --trade-rate-penalty 0.2 --target-trade-rate 0.15 --trade-rate-window 240 \
  --min-hold-steps 2 --trade-cooldown-steps 1 --max-trades-per-window 120 --trade-window-steps 240 \
  --constraint-violation-penalty 0.1
```

Result:
- Test return: **-0.1764%**
- Test trades: **2042**

Conclusion:
- Best result under MEXC standard fees in Exp 8.
- Trade count reduced by about 49% vs baseline (4034 -> 2042).

### Exp 8F - Min-hold only
Result:
- Test return: **-0.4855%**
- Test trades: **3884**

Conclusion:
- Single min-hold constraint alone did not improve enough.

### Exp 8G/H/I - MEXC-MX discounted fee regime checks

#### Exp 8G (constraints from 8E, fee_regime=mexc_spot_mx)
- Test return: **-0.1445%**

#### Exp 8H (8G + lower budget + fewer max positions)
- Test return: **-0.2752%**

#### Exp 8I (unconstrained legacy + mx fees + max_budget_per_trade=8)
- Test return: **-0.1414%**  *(best absolute PnL in Exp 8)*

#### Exp 8I full confirmation (50k)
- Test return remains **-0.14%** (stable)
- Test trades: **4033**

#### Exp 8I robustness (30 days / 5 symbols)
- Test return: **-0.3458%**
- Test trades: **8641**

Conclusion:
- Slightly better than MEXC-standard runs, but still negative.
- Robustness degrades on broader data.

---

### Experiment 8 summary
1. Hard constraints were implemented successfully and materially reduced churn when tuned mildly.
2. Aggressive constraints caused severe undertrading or no-trade collapse.
3. Best Exp 8 candidate remains mildly negative:
   - **Exp 8I (14d/3 symbols): -0.1414%**
4. On robustness data (30d/5 symbols), best candidate regressed to **-0.3458%**.
5. Net: Exp 8 improved fee realism and behavior control, but did not yet produce consistent profitability under realistic MEXC assumptions.

---

## Experiment 9 Campaign (April 2026) - Profitability + Broader-Data Robustness

### Goal
Improve net return while holding up on broader data slices (30d/5 symbols), instead of optimizing only 14d/3 symbols.

### New mechanics enabled
Added in `rl_trading_env.py` and exposed in `rl_trader.py`:

- `reward_equity_delta_scale`: rewards per-step equity growth relative to initial cash.
- `turnover_penalty_rate`: penalizes excessive traded notional.
- `continuous_drawdown_penalty`: smooth drawdown-aware penalty (not only threshold breach).
- `randomize_episode_start`, `min_episode_steps`, `max_episode_steps`: train on randomized windows.
- `fee_randomization_pct`: domain randomization for fee sensitivity.
- `--seed` for reproducible multi-seed comparisons.

Important implementation detail:
- Eval environments automatically disable randomized episode windows and fee jitter to keep validation/test deterministic and comparable.

### Exp 9 baseline command (single run)
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 10000 --mode backtest_3way --device cpu \
  --fee-regime mexc_spot_mx --action-mapping-mode legacy --seed 42 \
  --trade-action-bonus 1.2 --inactivity-penalty -0.35 --trade-execution-penalty 0.08 \
  --trade-rate-penalty 0.35 --target-trade-rate 0.14 --trade-rate-window 240 \
  --reward-equity-delta-scale 120 --turnover-penalty-rate 0.14 --continuous-drawdown-penalty 6.0 \
  --min-hold-steps 3 --trade-cooldown-steps 1 --max-trades-per-window 96 --trade-window-steps 240 \
  --constraint-violation-penalty 0.25 \
  --randomize-episode-start --min-episode-steps 1200 --max-episode-steps 3600 --fee-randomization-pct 0.20
```

### Exp 9 multi-seed + robustness sweep
Use the dedicated runner:

```bash
python run_experiment9.py
```

This executes both:
- core: `14d / 3 symbols`
- robust: `30d / 5 symbols`

for seeds `[11, 23, 47]`, and writes `exp9_results.json` plus per-run logs.

### Acceptance gate for Exp 9
Promote candidate only if all are true:

1. Median test return on `14d/3 symbols` > **0.00%** across seeds.
2. Median test return on `30d/5 symbols` >= **-0.10%** (better robustness floor).
3. No collapse modes:
   - not `0` trades,
   - and not near-max churn (`~1 trade/step`) on both splits.

---

## Experiment 9B (April 2026) - Positive Val + Test Candidate

### Goal
Address the observed Exp9 issue where validation could be profitable while test stayed negative.

### Exp 9B candidate (50k confirm settings)
```bash
python rl_trader.py --days 14 --max-symbols 3 --train-steps 50000 --mode backtest_3way --device cpu \
  --seed 11 --fee-regime mexc_spot_mx --transaction-cost 0.00003 \
  --action-mapping-mode legacy --invalid-sell-mode force_buy \
  --trade-action-bonus 15 --inactivity-penalty -5 --max-budget-per-trade 7 \
  --trade-rate-penalty 0.2 --target-trade-rate 0.15 --trade-rate-window 240 \
  --min-hold-steps 2 --trade-cooldown-steps 1 --max-trades-per-window 120 \
  --trade-window-steps 240 --constraint-violation-penalty 0.1
```

### 14d/3-symbol confirmation (50k, seeds 11/23/47)
Using `run_experiment9b.py` (`g_profit_both` variant):

- Median Val return: **+0.02%**
- Median Test return: **+0.10%**
- Median min(Val, Test): **+0.02%**
- Trades: ~**2041 / 2043** (Val/Test), no no-trade collapse

Artifacts:
- `exp9b_candidate_confirm.json`
- `exp9b_gate_g_profit_both_seed11.log`
- `exp9b_gate_g_profit_both_seed23.log`
- `exp9b_gate_g_profit_both_seed47.log`

### 30d/5-symbol robustness check (10k, seeds 11/23/47)
- Median Val return: **-0.00%**
- Median Test return: **-0.06%**

Interpretation:
- Candidate achieves the requested **positive Val + Test** behavior on core 14d/3-symbol runs.
- Robustness is improved versus earlier broad-data negatives but remains slightly below zero at median test on 30d/5-symbol.

### Notes
1. This candidate relies on an explicit fee override (`--transaction-cost 0.00003`), which is more optimistic than standard MEXC spot assumptions.
2. Next step is an Exp9B-realistic track: keep the same structural settings but move fee assumptions back toward `mexc_spot_mx` defaults and recover positive returns via policy/constraint tuning.

