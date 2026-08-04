# Crypto RL Agent — Optimization Plan

**Run analyzed:** `logs/run-20260803-083221-minimal`  
**Date:** 2026-08-03  
**Config:** SAC · `--rows 60000` · `--timesteps 200000` · `--window-size 60` · continuous action space

---

## 1. Observed Behavior Summary

### Eval episode outcomes (6 multi-seed evaluations)

| Episode file | Steps | Final PV | Return | Sell/Buy ratio | HOLD% |
|---|---|---|---|---|---|
| `ep1_1785748146` | 11 368 | $101.38 | **+1.38%** | 2.91× | **0%** |
| `ep1_1785748151` | 57 674 | $84.95 | **−15.05%** | 2.24× | **0%** |
| `ep1_1785748159` | 58 122 | $98.40 | **−1.60%** | 0.50× | **0%** |
| `ep1_1785748167` | 57 733 | $98.99 | **−1.01%** | 2.14× | **0%** |
| `ep1_1785748176` | 57 737 | $114.15 | **+14.15%** | 2.49× | **0%** |
| `ep1_1785748184` | 57 711 | $97.29 | **−2.71%** | 2.34× | **0%** |

**Multi-seed mean return: −0.81%** (excluding the first short 11k-step episode).  
**Negative reward steps across all episodes: 82–99%.**

### Training trajectory

- Final portfolio value mid-training oscillates between ~$91–$102 — the agent does not consistently grow capital.
- Training episodes show consistent −8% to −15% returns per episode (early episodes worse, converging to −7 to −8%).
- Entropy coefficient (`ent_coef`) is **frozen at 0.007** (no auto-tuning); critic loss is **massive (947–2875)** and highly oscillating — hallmark of a non-converged SAC critic.
- Actor loss is negative and large (−52 to −71) — while expected in SAC, the high variance suggests unstable policy updates.

---

## 2. Identified Weaknesses (Ranked by Impact)

### W1 — Degenerate action collapse: ZERO HOLD actions ⚠️ CRITICAL

**Observation:** In every single eval and training episode, `HOLD` count = 0 (0%). The agent trades on **every single minute bar**, alternating BUY and SELL with ~70% sell bias. Average buy amount is 85–100% of cash. Average sell amount is 94–100% of holdings.

**Root cause analysis:**

1. The `excess_return` reward implicitly penalizes holding (via `hold_cost_rate`), but the real culprit is `profit_bonus=2.91`. A flat +2.91 reward per profitable sell is enormous relative to the base signal (±0.001–0.01 per step). This creates a hyperactive "churn" incentive: buy → sell immediately → collect bonus → repeat.
2. The continuous action space uses `threshold=0.05`. Since the policy output virtually never falls within ±5% of zero (the SAC entropy pushes it toward the extremes), every step generates a trade on at least one asset.
3. `empty_buy_penalty=0.059` and `empty_sell_penalty=0.042` penalize zero-amount trades, but these only apply to the MultiDiscrete path — they are **dead code** in the continuous action path actually used here.

**Impact:** The agent is a **fee-burning flip machine**, not a trading agent. It incurs 0.10–0.23% in fees per episode on top of negative alpha — trading ~57,000 actions across 9 assets per 6,458-step episode (about 9 trades per bar).

---

### W2 — Massive unstable SAC critic loss ⚠️ HIGH

**Observation:** SAC critic loss ranges from **947 to 2875** with no clear downward trend across the full 200k-step run.

**Root cause analysis:**

1. The large, occasional `profit_bonus` spikes (+2.91 per sell) create a **non-stationary reward distribution**. SAC's Q-function bootstraps from these spikes, causing target values to diverge.
2. With `gamma=0.9944` and episodes of ~6,000–7,000 steps, the effective horizon is `1/(1−0.9944) ≈ 179 steps`. A +2.91 bonus at step 50 propagates a discounted cumulative signal of ~520 back through the rollout, completely overwhelming the per-step market signal of ~0.001.
3. `ent_coef=0.007` passed as a fixed float to SB3's `SAC()` **disables auto-tuning** — `ent_coef_loss = 0.0000` across the entire run confirms this. The agent has no adaptive entropy regulation.

---

### W3 — Reward signal drowned by shaping noise ⚠️ HIGH

**Observation:** The `excess_return` signal at 1-minute resolution is ±0.001–0.01. Shaping terms (`profit_bonus=2.91`, `terminal_return × 5.0`) are three orders of magnitude larger.

**Root cause analysis:**

1. Multiple reward components with wildly different magnitudes make it impossible for the critic to learn a stable value function. The effective signal-to-noise ratio is essentially zero.
2. The Optuna objective (mean multi-seed portfolio value) optimizes a noisy surrogate. One lucky seed can dominate the average, causing the search to over-fit to a specific regime rather than finding a robust policy.
3. The `* 10.0` scaling on the base excess return (in `env.py:610`) partially compensates but is still outweighed by the bonus.

---

### W4 — Agent never learns patience (no hold) MEDIUM

**Observation:** Even with `hold_cost_rate=0.000002` (≈ 0.12% per day), the policy never produces actions within the ±0.05 dead zone.

**Root cause analysis:**

1. SAC entropy pushes the action distribution toward a flat/uniform distribution over [−1, 1]. The dead zone occupies only 5% of the action space and is actively suppressed by Q-gradients that reward trading.
2. `max_single_step_allocation=0.5` is not enforced when the per-asset action `< 0.5` and `sum_buys ≤ 1.0`. With 9 assets each buying 30–90%, the normalization path that would cap individual buys is bypassed.

---

### W5 — Very high cross-market variance MEDIUM

**Observation:** Returns range from −15.05% to +14.15% across 6 seeds on the same policy — a 29pp spread.

**Root cause analysis:**

1. The policy has converged to a **momentum-following strategy**: buy on up-moves, sell on down-moves. This works in trending markets and fails catastrophically in choppy/ranging regimes.
2. 200k steps on a 48k-step training window (≈33 days) is insufficient to see a full crypto market cycle. The agent overfits to the specific regime in its training window.
3. With `data_seed=42` fixed, each training episode samples from the same 60k-row pool, exposing the agent to a limited slice of the 4-year dataset.

---

### W6 — Observation window too narrow for meaningful alpha LOW-MEDIUM

**Observation:** `window_size=60` (60 minutes) with RSI(14), MACD, and momentum all computed over that window.

**Root cause analysis:**

1. 60-minute lookback captures micro-structure noise but misses the daily/weekly cycles that drive most crypto alpha.
2. No volume information is included — volume confirmation is one of the most robust short-term predictors in crypto.
3. All technical indicators (RSI, MACD, momentum) are normalized over the same rolling 100-bar z-score window, reducing their discriminative power.

---

### W7 — Policy network too small for observation space LOW

**Observation:** `[128, 128]` actor/critic nets; obs dim = `(60+5)×9 + 2 + 1 + 3×9 = 615`.

**Root cause analysis:**

1. A 615→128→128 bottleneck is extremely aggressive, discarding most raw input signal immediately.
2. Trained on 200k environment steps — the effective sample count per parameter is reasonable, but the information loss in the first layer is irreversible.

---

## 3. Optimization Plan

### Priority 1 — Fix the Degenerate Trading Loop (1–3 days)

These are code-level changes that must be applied before any further HPO, as the current regime makes all Optuna results unreliable.

#### 1A. Scale `profit_bonus` relative to trade notional (not flat)

The flat bonus of 2.91 overwhelms the market signal. Replace with a proportional bonus:

```python
# env.py, in the continuous sell block (currently line ~467):
# BEFORE:
profit_bonus_reward += self.profit_bonus

# AFTER:
# Scale bonus to actual % profit, so a 1% win gives +profit_bonus reward:
pct_profit = asset_realised_pnl / (t_units * t_price)  # e.g. 0.01 for 1%
profit_bonus_reward += self.profit_bonus * pct_profit
```

**Config change:** Re-run Optuna with `profit_bonus` in `[0.0, 0.5]`.

#### 1B. Widen the continuous action dead zone and expose as CLI parameter

```python
# env.py step(), change:
threshold = 0.05
# to something configurable:
threshold = self.action_dead_zone  # default 0.15

# Also add a micro-incentive for each asset that stays in the dead zone:
n_held = sum(1 for i in range(self.num_assets) if abs(action[i]) <= threshold)
reward += n_held * self.hold_incentive  # e.g. 0.0005 per held asset
```

Expose `action_dead_zone` and `hold_incentive` as CLI arguments and include them in the Optuna search.

#### 1C. Fix SAC entropy auto-tuning

Passing a fixed float disables auto-tuning in SB3. Change `main.py`:

```python
# BEFORE:
ent_coef=ent_coef if ent_coef != 0.01 else "auto",

# AFTER:
ent_coef="auto",  # always auto-tune; removes one confounded hyperparameter
```

Remove `--ent-coef` from `run_large.sh` and the Optuna search.

---

### Priority 2 — Stabilize the Reward Signal (Week 1)

#### 2A. Add reward clipping via VecNormalize

Wrap the training environment:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
train_env = VecNormalize(
    DummyVecEnv([lambda: raw_train_env]),
    norm_reward=True,
    norm_obs=False,  # obs already normalized manually
    gamma=args.gamma,
    clip_reward=5.0,
)
```

This prevents large reward spikes from destabilizing the critic while preserving relative reward ordering.

#### 2B. Reduce terminal bonus magnitude

```python
# env.py line ~636, change:
reward += terminal_return * 5.0
# to:
reward += terminal_return * 1.0
```

With 6,000-step episodes and ±15% returns, the 5× multiplier can add ±0.75 to the terminal step — larger than the cumulative discounted signal from 100 prior steps.

#### 2C. Switch Optuna objective to Sharpe ratio

```python
# scripts/optuna_search.py, in run_experiment() or objective():
pv_series = np.array([v["value"] for v in eval_portfolio_values])
returns = np.diff(pv_series) / pv_series[:-1]
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(525600)  # 1-min annualized
return float(sharpe)
```

Sharpe ratio penalizes variance across regimes, making HPO more robust than optimizing raw portfolio value.

---

### Priority 3 — Broaden Optuna Search Space (Weeks 2–3)

Current Optuna only tunes 3 parameters (`profit_bonus`, `hold_cost_rate`, `illegal_buy_penalty`).  
After Priority 1–2 fixes, expand to the full set below:

| Parameter | Current | Suggested Search Range | Notes |
|---|---|---|---|
| `profit_bonus` | 2.91 | `[0.0, 0.5]` | After switch to % bonus |
| `hold_cost_rate` | 0.000002 | `[1e-7, 1e-5]` (log) | Fine range |
| `learning_rate` | 0.00002 | `[5e-5, 5e-4]` (log) | SAC often benefits from higher LR |
| `gamma` | 0.9944 | `[0.98, 0.999]` | Effective horizon sensitivity |
| `batch_size` | 256 | `{128, 256, 512, 1024}` | Replay sample diversity |
| `action_dead_zone` | 0.05 | `[0.05, 0.30]` | New parameter (Priority 1B) |
| `hold_incentive` | 0.0 | `[0.0, 0.002]` | New parameter (Priority 1B) |
| `n_steps` | 1024 | `{512, 1024, 2048, 4096}` | Replay buffer fill rate |
| `window_size` | 60 | `{30, 60, 120, 240}` | Lookback sensitivity |

**Target:** 200 trials at `--rows 40000 --timesteps 150000`, using `n_jobs=2–4` if hardware permits.

---

### Priority 4 — Improve Data Regime Coverage (Week 3)

#### 4A. Audit the episode re-sampling pipeline

`train_env` is correctly set up with `parquet_path` and `n_rows`, enabling random window re-sampling each episode. However, `read_last_n` scans the full parquet on every `reset()` call (38 times in the analyzed run). At 60k rows × 9 symbols, this is slow. 

**Fix:** Pre-load all valid start timestamps once at startup and sample from the cached list in `reset()`.

#### 4B. Increase training data window

With 4 years of data available, `--rows 60000` exposes the agent to only ~42 days per episode. Increase to:

- `--rows 200000` (~140 days, ~800 MB RAM)
- `--rows 400000` (~280 days, ~1.6 GB RAM)

Larger windows force the agent to see bull, bear, and ranging regimes within a single episode, dramatically reducing policy variance.

#### 4C. Add BTC regime features explicitly

The `btc_mom_norm` and `btc_vol_norm` signals are already computed but currently occupy only 2 of 617 observation dimensions. Consider:

1. Placing them at the **start** of the observation vector (not buried at the end), so the first network layer can assign them large weights without fighting random initialization.
2. Adding a **trend regime label** (1-hot or soft): bull / ranging / bear, derived from a simple 24h/7d momentum comparison.

---

### Priority 5 — Architectural Changes (Month 2)

#### 5A. LSTM / Recurrent policy

The current MLP policy has no temporal memory. Each step is processed independently, which is fundamentally mismatched with sequential financial data. Adding a recurrent layer can:

- Capture multi-step patterns (e.g., accumulation → breakout) that are invisible to a memoryless policy
- Implicitly track position state without relying solely on the portfolio observation features

**Implementation path:**
```python
# Install sb3-contrib:
pip install sb3-contrib

from sb3_contrib import RecurrentPPO
model = RecurrentPPO(
    "MlpLstmPolicy",
    train_env,
    n_steps=2048,
    batch_size=128,
    policy_kwargs={"lstm_hidden_size": 128, "n_lstm_layers": 1},
)
```

Note: This requires switching from SAC to PPO (RecurrentPPO). The trade-off is lower sample efficiency from PPO but better temporal modeling.

#### 5B. Portfolio simplex action space

Replace the per-asset continuous action vector with a portfolio allocation simplex:

```
action ∈ ℝ^{N+1}, normalized via softmax → target weights (sum = 1.0)
reward = Σ_i w_i * r_i − transaction_cost(|Δw|)
```

**Advantages:**
- Eliminates illegal buy/sell states entirely (no negative allocations possible)
- Removes the need for `illegal_buy_penalty`, `illegal_sell_penalty`, `empty_buy_penalty`, `empty_sell_penalty`
- Makes position sizing directly interpretable
- Transaction cost is naturally proportional to how much the portfolio rebalances

**Implementation:** Replace `Box(-1, 1, shape=(N,))` with `Box(0, 1, shape=(N+1,))`, add a `softmax` normalization in `step()`, and compute rebalancing cost as `fee_rate × Σ |new_weight - old_weight| × portfolio_value / 2`.

#### 5C. Add volume and OHLC features

The current environment only uses `close` prices. Adding volume is the single highest-value free feature in crypto:

1. Load `open`, `high`, `low`, `volume` columns in `data.py`
2. Compute normalized volume: `vol / vol.rolling(W).mean() - 1`
3. Compute intrabar volatility: `(high - low) / close`
4. Compute VWAP deviation: `close / (Σ(close × volume) / Σ(volume)) - 1`

This adds 3×N new features to the observation, requiring a network size bump from `[128, 128]` to at least `[256, 256]`.

#### 5D. Multi-timeframe observation stack

Rather than one 60-minute window at 1-minute resolution, compute indicators at multiple resolutions:

| Timeframe | Window | Features | Purpose |
|---|---|---|---|
| 1-min | 30 bars | Price changes, RSI | Micro-structure momentum |
| 5-min | 24 bars | Price changes, Vol | Hourly drift |
| 60-min | 24 bars | Price changes, Trend | Daily structure |

This provides a 3× longer effective lookback without 3× more raw features, and allows the network to learn to combine signals across scales.

---

### Priority 6 — Training Infrastructure (Ongoing)

#### 6A. Walk-forward (temporal) train/test split

The current `read_train_test()` samples a random window from the full dataset. This can cause the test window to be temporally earlier than training data, enabling subtle lookahead leakage via the indicator pre-calculations.

**Fix:** Always hold the **last 20% of chronological time** as the test set:
```python
# In main.py:
# Load the last (n_train + n_test) rows chronologically,
# then split at the 80% mark — never resample the test window.
```

#### 6B. Parallel environments (VecEnv)

```python
n_envs = 4  # or more if RAM permits
train_env = DummyVecEnv([
    lambda: MinimalCryptoEnv(..., parquet_path=args.parquet_path, n_rows=args.rows)
    for _ in range(n_envs)
])
```

With episode re-sampling enabled, each of the 4 environments independently draws a different random market window per episode, providing ~4× more diverse training data per policy update at minimal CPU cost.

#### 6C. Save model checkpoints for best Sharpe

Add a `CheckpointCallback` that saves the model whenever a new best Sharpe is achieved on the test set, not just at fixed intervals. This prevents the final model from being worse than an intermediate checkpoint due to training instability.

---

## 4. Prioritized Action Roadmap

| Phase | Key Actions | Effort | Expected Impact |
|---|---|---|---|
| **Immediate (days 1–3)** | 1A: Scale profit_bonus; 1B: widen dead zone + hold incentive; 1C: SAC auto entropy | Low code changes | Eliminate 0% HOLD regime |
| **Week 1** | 2A: VecNormalize; 2B: reduce terminal bonus; 2C: Sharpe as HPO objective | Medium | Stable critic, reliable HPO |
| **Weeks 2–3** | 3: Optuna expanded search (200 trials, full parameter set) | Compute | Find robust hyperparameter region |
| **Week 3** | 4A: Cache reset timestamps; 4B: increase rows to 200k; 4C: BTC regime features | Medium | Broader regime coverage |
| **Month 2** | 5A: LSTM/RecurrentPPO; 5B: Portfolio simplex; 5C: Volume features; 5D: Multi-timeframe | High | Architectural alpha uplift |
| **Ongoing** | 6A: Walk-forward split; 6B: VecEnv parallel; 6C: Sharpe-checkpoint | Medium | Robust evaluation and training |

---

## 5. Key Open Questions

1. **What market regime was the training data in?** `state.json` shows `buy_hold_baseline=$98.38` (BTC down ~1.6% in the eval window). If training also covered a bear/choppy period, the sell-heavy policy is a direct consequence of regime fitting, not a general failure. Knowing the date range of the 60k-row training window would clarify this.

2. **GPU availability?** The current SAC with a 615-dim obs and `[128, 128]` network is fast on CPU. Priority 5 changes (LSTM, larger nets) would benefit significantly from GPU training.

3. **Target deployment?** If this is for live Binance Spot trading:
   - The simulated fee rate (`0.001` = 0.1%) is **2× the real taker fee** (0.05%). This makes the agent overly conservative about trading — consider switching to 0.0005.
   - Slippage is not modeled at all. For small positions (<$10k), 1-min close execution is a reasonable approximation. For larger positions, bid-ask spread and market impact need to be added.

4. **Acceptable drawdown and Sharpe threshold?** The eval episodes show max drawdowns of 4.8% to 17.3% on ~6k-step windows (~4.5 days). What is the acceptable max drawdown for a live deployment? This determines the appropriate `hold_cost_rate` and whether a stop-loss mechanism is needed.

---

## 6. Quick Wins (Ship Today)

These three changes require no architectural modification and can be validated with a single run:

```bash
# In main.py, change the SAC ent_coef line:
#   FROM: ent_coef=ent_coef if ent_coef != 0.01 else "auto",
#   TO:   ent_coef="auto",

# Then launch with reduced profit_bonus and adjusted penalties:
python main.py \
  --parquet-path binance_spot_1m_last4y_single.parquet \
  --rows 60000 --timesteps 200000 --window-size 60 \
  --data-seed 42 --dashboard True \
  --reward-type excess_return \
  --fee-rate 0.001 \
  --hold-cost-rate 0.000002 \
  --profit-bonus 0.3 \
  --empty-buy-penalty 0.001 \
  --empty-sell-penalty 0.001 \
  --budget-initial 100.0 \
  --gamma 0.9944 --batch-size 256 \
  --illegal-buy-penalty 0.001 \
  --learning-rate 0.00003 \
  --n-steps 1024 --clip-range 0.2 \
  --algorithm SAC \
  --illegal-sell-penalty 0.001 \
  --max-single-step-allocation 0.5

# Success signal: HOLD% > 0 in any eval episode.
# Failure signal: HOLD% still = 0% → implement Priority 1B (widen dead zone in code).
```
