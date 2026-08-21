# Crypto RL Trading Bot — Systematic Analysis & Optimization Plan

> **Run analysed:** `run-20260821-080040-minimal` (1.2M timesteps, PPO/RecurrentPPO, 400k rows)
> **Analysis date:** 2026-08-21

---

## 1. Executive Summary of Current Results

| Metric | Value | Verdict |
| -------- | ------- | --------- |
| Final eval portfolio | **$88.68** (started $100) | ❌ Loss |
| Eval PnL | **–$11.33** | ❌ |
| Buy-and-hold baseline | **$98.36** | ❌ Agent underperforms |
| Eval Sharpe | **–8.90** | ❌ Catastrophic |
| Eval win rate | **0.0%** | ❌ |
| Training portfolio (ep4) | **$0.24** (started $100) | ❌ Total wipeout |
| Drawdown at end of training | **–99.75%** | ❌ |
| Clip fraction | **0.0** for all 1.2M steps | ❌ Policy not learning |
| Approx KL | ~3e-8 (stable near zero) | ❌ Policy not updating |

**The agent is not learning to trade profitably.** It generates a ~$0.24 training portfolio by the end of training and a –11% eval loss. Despite these catastrophic numbers, the evaluator assigns a `profit_bonus` of **0.5 per step** on 100% of evaluation steps — a critical reward design bug that masks the true failure.

---

## 2. Critical Bugs (P0) — Fix These First

### 2.1 🐛 `profit_bonus` is awarded on **every single continuous-action step**

**Location:** [`env/minimal_env.py` L346–347](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/minimal_env.py#L346-L347)

```python
else:  # continuous action space
    reward_components["profit_bonus"] = self.profit_bonus  # Always 0.5, every step!
```

**Evidence from eval log:**

| Component | Cumulative total |
| ----------- | ----------------- |
| `profit_bonus` | **+4,384** (constant 0.5 × 8,769 steps) |
| `drawdown_penalty` | –3,067 |
| `market_alpha` | –2.6 |

The `profit_bonus` is **1,000× larger than the actual performance signal** and is completely independent of trade quality. The agent is rewarded the same for profitable and loss-making trades. The meaningful `market_alpha` signal is buried in noise.

**Fix:** Gate on realized alpha vs. hold:

```python
# Only reward when portfolio genuinely beats market after fees:
if portfolio_return > market_return + (2 * self.fee_rate):
    reward_components["profit_bonus"] = self.profit_bonus * (
        portfolio_return - market_return
    )
```

---

### 2.2 🐛 Policy has completely collapsed — clip_fraction = 0, KL ≈ 3e-8

**Evidence:**

```
clip_fraction: 0.0 (every measured step, all 1.2M timesteps)
approx_kl:     ~3e-8 (essentially zero, never changes)
policy_loss:   –9e-6 (near zero, constant)
value_loss:    ~0.7 (rising, not converging)
```

The PPO policy gradient is doing **nothing**. The learning rate of `8e-8` is at least **3,000× too small** for RecurrentPPO with these reward scales. SB3's default is `3e-4`. The clip fraction of 0.0 means no constraint is ever active — the policy parameters are frozen.

**Fix:**

```bash
--learning-rate 0.0001  # was 8e-8 — increase by ~1250×
```

---

### test

### 2.3 ✅ Observation dimension mismatch between `__init__` and `feature_utils` [FIXED]

#### Root Cause

- `feature_utils.py` computes `static_dim = macro_dim (5) + 7 * N` for the 7 statistical indicators pre-computed in `precalc_static_obs`.
- `observation.py` builds the observation vector dynamically:
  - (1) Macro features: `5`
  - (2-5) Dynamic time-series windows: `(W + 30 + 24 + 24) * N = (W + 78) * N`
  - (6) Statistical indicators from `precalc_static_obs`: `7 * N`
  - (7) Dynamic portfolio features: `1 (cash_pct) + N (holdings_pct) + N (unrealised_pnl_buf) + N (has_pos) + 1 (drawdown) = 2 + 3 * N`
- Total obs buffer dimension is `5 + (W + 78) * N + 7 * N + 2 + 3 * N`.
- `MinimalCryptoEnv.__init__` previously overloaded `self.static_per_asset_dim = window_size + 7 + 78`, which misrepresented the pre-computed static feature dimension in `precalc_static_obs` and became inconsistent when `precalculate_static_obs(self)` set `env.static_per_asset_dim = 7`.

#### Resolution

- Updated [minimal_env.py](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/minimal_env.py) `__init__` to define `self.static_per_asset_dim = 7` (matching `feature_utils.py` and `precalc_static_obs`), compute `dynamic_windows_dim = (window_size + 78) * self.num_assets`, and construct `obs_dim` cleanly.
- Verified that `obs_buf`, `observation_space`, and `precalc_static_obs` dimensions match exactly across initialisation and environment resets.

---

### 2.4 🐛 Training portfolio wipes out to $0.24 due to fee bleeding

Training ep4: **13,655 SELLs + 13,598 BUYs, only 71 HOLDs** out of ~27,000 steps. Each trade incurs a 0.1% fee. The softmax continuous rebalancer applies turnover fees every minute, destroying capital. The agent learned to churn because `profit_bonus` rewards every step equally.

---

## 3. Major Design Weaknesses (P1)

### 3.1 ✅ Continuous action = full portfolio rebalance every minute → impossible fees [FIXED]

`apply_continuous_action` previously rebalanced the portfolio fully on every step regardless of turnover size. With `fee_rate = 0.001` and typical 10% turnover, this cost 0.05%/minute = 72%/day in fees.

#### Resolution of 3.1

- Added a `min_turnover_threshold` check in [`apply_continuous_action`](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/action_processing.py):

  ```python
  turnover = np.sum(np.abs(target_weights - old_weights))
  min_turnover_threshold = getattr(env, "min_turnover_threshold", 0.02)
  if turnover < min_turnover_threshold:
      return 0.0, 0.0
  ```

- Exposed `min_turnover_threshold: float = 0.02` as a configurable parameter in [`MinimalCryptoEnv`](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/minimal_env.py) and CLI argument `--min-turnover-threshold` in [`crypto_rl/cli.py`](file:///home/konstantin/dev/crypto_rl/crypto_rl/cli.py).

---

### 3.2 ⚠️ Reward components are wildly misscaled

| Component | Effective range/step | Relative magnitude |
| ----------- | -------------------- | -------------------- |
| `profit_bonus` | +0.5 (constant) | **Dominant** |
| `drawdown_penalty` | 0 to –7.72×drawdown | Large |
| `market_alpha` | ±0.01 (typical 1-min return) | **1000× smaller** |
| `hold_cost` | –4.5e-5 | Negligible |

The agent optimizes the constant `profit_bonus` and tries to minimize drawdown. It never receives a useful market signal.

**Fix:** Set `profit_bonus = 0`, increase `base_penalty` from 10 to 100, reduce `drawdown_penalty_coef` from 7.72 to 0.5:

```bash
--profit-bonus 0.0
--drawdown-penalty-coef 0.5
```

---

### 3.3 ⚠️ `gamma = 0.9863` → only 73-minute effective horizon

At 1-min bars: `1/(1-0.9863) ≈ 73 steps`. Any trend or position held longer than ~1 hour is invisible to the agent. For meaningful trend-following:

```bash
--gamma 0.999   # ~1000 steps = ~17 hours effective horizon
```

---

### 3.4 ⚠️ Optuna search space is centered on the wrong region

From `optuna_search.py`:

```python
args.learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-7, log=True)  # too low
args.gamma = trial.suggest_float("gamma", 0.98, 0.9999)                           # OK
args.profit_bonus = trial.suggest_float("profit_bonus", 0.25, 1.0)               # should be 0
args.drawdown_penalty_coef = trial.suggest_float("drawdown_penalty_coef", 5, 20) # too high
```

The LR range `[1e-10, 1e-7]` will always produce a collapsed policy. The "best" found LR of `~1e-8` is the best among broken configurations.

**Fix:**

```python
args.learning_rate = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
args.profit_bonus = trial.suggest_float("profit_bonus", 0.0, 0.01)
args.drawdown_penalty_coef = trial.suggest_float("dpc", 0.1, 2.0)
# Fix broken fallback score:
except Exception as e:
    return -float("inf")   # was 100.0 — bad trials polluted the study
```

---

### 3.5 ⚠️ `_rolling_zscore` is O(T²) — computation bottleneck

The feature precomputation function iterates over T rows in a Python loop, doing O(T) numpy work per row. At T=400,000 this is ~1.6×10¹¹ operations. This makes every episode reset extremely slow and limits the number of Optuna trials that can be run.

**Fix:** Vectorize using the cumulative sum trick (identical to `_rolling_mean_std`):

```python
def _rolling_zscore_fast(arr, window=100):
    cs = np.cumsum(arr, axis=0)
    cs2 = np.cumsum(arr**2, axis=0)
    # ... vectorized O(T) computation
```

---

### 3.6 ⚠️ `RecurrentPPO` `policy_kwargs` architecture spec is silently ignored

In `experiment.py`:

```python
policy_kwargs = {
    "net_arch": dict(pi=[128, 128], qf=[128, 128]),  # ← ignored by MlpLstmPolicy!
    ...
}
# But then for RecurrentPPO:
model = RecurrentPPO(
    "MlpLstmPolicy", ...,
    policy_kwargs={"lstm_hidden_size": 128, "n_lstm_layers": 1},  # overwrites above
)
```

The `net_arch` dict is defined for SAC but never applied to RecurrentPPO.

---

## 4. Secondary Issues (P2)

| Issue | Impact | Fix |
| ------- | -------- | ----- |
| Train/test is always the most recent N rows — no temporal generalization | Medium | Walk-forward validation |
| `--skip-multi-seed-eval` flag — overfitting undetected | Medium | Remove flag |
| Win-rate metric in callback counts steps, not trades | Low | Fix counter logic |
| `empty_buy_penalty = 0.073 × portfolio` = $7.3 per empty BUY | Medium | Reduce to 0.001 |
| Single-asset data file despite multi-asset code | Medium | Verify data file |
| Checkpoint keeps "best Sharpe" but Sharpe is catastrophically negative | Low | Use portfolio_value threshold |

---

## 5. Optimization Plan (Prioritized)

### 🔴 P1 — Fix reward design (~2 hours)

1. Remove constant `profit_bonus` for continuous actions:

   ```python
   # Delete the unconditional assignment; gate on positive alpha
   ```

2. Rescale: `base_penalty = 100.0`, `drawdown_penalty_coef = 0.5`
3. Amplify terminal signal: `terminal_reward = terminal_return * 10.0`

### 🔴 P2 — Fix learning rate and key hyperparameters (~30 min)

```bash
--learning-rate 0.0001    # was 8e-8
--gamma 0.999             # was 0.9863
--ent-coef 0.01           # was 0.001
```

### 🔴 P3 — Add turnover threshold OR switch to discrete actions (~1 hour)

```python
# In apply_continuous_action:
if turnover < 0.02:
    return 0.0, 0.0  # skip rebalance
```

### 🟠 P4 — Fix Optuna search space and objective (~2 hours)

- LR range: `1e-5` to `3e-4`
- profit_bonus range: `0` to `0.01`
- Fix fallback to `-inf` not `100.0`
- Consider Calmar ratio as objective instead of Sharpe

### 🟠 P5 — Vectorize `_rolling_zscore` (~2 hours)

Replace O(T²) Python loop with vectorized cumsum approach.

### 🟡 P6 — Fix RecurrentPPO architecture + consider SAC (~4 hours)

- Fix `policy_kwargs` for LSTM (add `net_arch` layers)
- Evaluate SAC for better sample efficiency on continuous actions

### 🟡 P7 — Walk-forward validation + re-enable multi-seed eval [FIXED]

- Implemented 3-fold expanding walk-forward cross-validation (`get_walk_forward_splits` and `--cv-folds`).
- Preserves chronological order across folds and aggregates out-of-sample performance metrics.
- Re-enabled multi-seed evaluation with `--skip-multi-seed-eval` flag available for fast smoke tests.

### 🟢 P8 — Radical alternatives (if P1–P7 insufficient, ~1-2 days)

- **Behavioural cloning pre-training** from momentum strategy
- **Potential-based reward shaping** using rolling Sharpe as Φ
- **Auxiliary price prediction head** to improve representation
- **Confirm/expand dataset** to multiple regime periods

---

## 6. Sanity Check — Expected Signals of a Healthy Run

After applying P1–P3, a 50k-timestep smoke test should show:

| Signal | Expected | Current (broken) |
| -------- | ---------- | ----------------- |
| `clip_fraction` | 0.01–0.10 | 0.0 ❌ |
| `approx_kl` | 0.001–0.01 | 3e-8 ❌ |
| `policy_loss` | varies ±0.01 | –9e-6 (static) ❌ |
| `value_loss` | decreasing | rising ❌ |
| `portfolio_value` (training) | stays ~$90-110 | collapses to $0 ❌ |
| `profit_bonus` (cumulative) | ≪ `market_alpha` | 1000× larger ❌ |

```bash
# Quick validation run:
./.conda/bin/python main.py --n-rows 20000 --timesteps 50000 \
    --n-envs 4 --action-space-type multidiscrete \
    --learning-rate 0.0001 --gamma 0.999 --ent-coef 0.01 \
    --profit-bonus 0.0 --drawdown-penalty-coef 0.5 \
    --dashboard
```
