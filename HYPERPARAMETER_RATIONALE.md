# Hyperparameter Rationale — `run_large.sh`

> **Purpose:** This document explains the *reasoning* behind each hyperparameter
> used in `scripts/run_large.sh`. It is intended as living documentation and as
> structured LLM input for diagnosing bot behaviour and identifying improvement
> opportunities. Update this file whenever you change a value in the launch
> script.

---

## 1. Hyperparameter Rationale

### Action Space

#### `--action-space-type multidiscrete`

`MultiDiscrete([3, num_assets, 101])` encodes action type (Hold/Buy/Sell),
target asset, and trade size as three independent integers. This is preferred
over the continuous `Box` alternative because it pairs naturally with
**action masking** (via `MaskablePPO` + `ActionMasker`): invalid action types
(e.g., SELL when holding nothing) can be zeroed out at the logit level before
sampling, giving a clean safety guarantee without relying solely on reward
penalties. The discrete representation also reduces the policy's output
dimension compared to per-asset continuous weights.

#### `--action-dead-zone 0.60`

In the multidiscrete action space, dimension 2 encodes trade size as an
integer 0–100 (mapped to 0.0–1.0). Any value below `action_dead_zone`
(i.e., amount_pct < 0.85) is silently converted to a **Hold** before the
action reaches the environment. The current value is higher
than the Optuna best-params default (0.50) to trade less frequently and reduce
the observed fee leakage.

#### `--max-asset-allocation 0.25`

Hard cap on the fraction of total portfolio value that can be held in any single
asset at any point in time. Enforced inside `apply_discrete_action` by comparing
the current exposure to the cap before executing a buy. At 25% this allows
up to 4 assets to hold equal-weight positions, providing a minimum level of
diversification while still allowing meaningful concentration bets. Combined
with 9 tradeable assets, the bot cannot "go all-in" on one position.

#### `--max-single-step-allocation 0.25`

Maximum fraction of available **cash** that can be spent in a single buy
action. This was recently raised from 0.15 (the default) to 0.25 to let the
agent build positions faster when it has high conviction — without this limit,
a high dead-zone and a lower cash-fraction cap would make meaningful buys
nearly impossible in early episodes.

#### `--min-turnover-threshold 0.10`

Active **only in continuous action mode** (`apply_continuous_action` in
[`action_processing.py`](crypto_rl/env/action_processing.py)). After the
softmax target weights are computed, the function calculates total portfolio
turnover as the L1 norm of the weight delta
(`turnover = Σ |target_weight − current_weight|`). If this value is below the
threshold, the entire rebalance is short-circuited — holdings and cash are left
unchanged and zero fees are charged. This prevents the continuous policy from
burning fees on near-zero portfolio shifts that carry no real directional signal.

At 0.10, the policy must intend to shift at least 10% of total portfolio weight
before a rebalance executes, which is a meaningful bar given the softmax output
can naturally produce small weight oscillations each step. The threshold is read
at runtime via `getattr(env, "min_turnover_threshold", 0.02)`, so it falls back
to 0.02 if not set (the original default when the parameter was first introduced).

**In multidiscrete mode this parameter has no effect.** `apply_discrete_action`
does not reference it; trade execution is gated entirely by the action dead zone
and the action masking instead.

---

### Reward Function

#### `--reward-type excess_return`

The agent is rewarded for its **return over the equal-weight market return**
(alpha), not raw PnL. This incentivises the bot to actually beat a passive
index rather than just riding a bull market. A 1.2× asymmetric multiplier
is applied to negative alpha to slightly penalise underperformance more than
outperformance is rewarded (see `minimal_env.py` step logic). Rewards are
clipped to [−1.0, +1.0] to prevent variance explosion.

#### `--profit-bonus 0.1`

A shaped reward bonus applied on every **sell** that closes a position with
positive realised PnL. The bonus is proportional to the trade return
(`profit_bonus × trade_return`), so large profitable trades are rewarded more
than small ones. At 0.1 this is a modest supplement to the main alpha signal;
it is intended to reduce the agent's tendency to hold losing positions
indefinitely.

#### `--drawdown-penalty-coef 0.15`

Coefficient applied to the **delta drawdown** each step. Only worsening
drawdown steps are penalised (i.e., when the portfolio moves further below its
peak); recoveries are not rewarded to avoid encouraging risk-seeking behaviour.
It was decreased to 0.15 to allow trades enough breathing room to develop, even through short-term price fluctuations.

#### `--hold-cost-rate 0.000001`

A micro-penalty (1e-6 per step) applied to any position whose unrealised PnL
is worse than −5%. At 1-min bars, 1e-6 × 1440 min/day ≈ 0.14% per day — a
gentle but persistent drag on deeply underwater positions. The previous default
(1e-4) was catastrophically high (~6%/hour) and caused the agent to immediately
liquidate all holdings, so the value was reduced dramatically. The current value
is intentionally tiny to avoid overpowering the main reward signal.

#### `--hold-incentive 0.0`

Micro-reward for assets sitting in the action dead zone (continuous mode only).
Disabled (0.0) for the current multidiscrete mode because the hold signal is
already implicit: the dead zone remaps low-conviction actions to Hold for free.

---

### Penalty Structure

> **Design note:** The environment does **not** implement per-asset-conditional
> action masks (i.e., masking out specific assets on a per-sell basis). The
> action mask only gates the top-level action type (Buy/Sell). Individual
> asset-specific illegal actions (e.g., selling an asset you don't hold) are
> handled via reward penalties and forced remapping to Hold. The penalties must
> therefore be large enough to train the policy away from these actions without
> conditional masking.

#### `--illegal-buy-penalty 0.1` and `--illegal-sell-penalty 0.1`

Penalty added to the step reward when the agent attempts to:

- **Illegal buy**: BUY when cash ≈ 0 (cannot afford anything)
- **Illegal sell**: SELL an asset with zero holdings (short-selling attempt)

At 0.1 these are equal to 200× the default `RULE_PENALTY` constant (5e-4),
making illegal actions strongly aversive. The rule_penalties cumulative value
was **0.0** in the last eval, confirming the policy has successfully learned to
avoid illegal actions — the penalties have done their job and could potentially
be reduced.

#### `--empty-buy-penalty 0.1` and `--empty-sell-penalty 0.1`

Penalty for selecting Buy or Sell with a zero `amount_pct` (after the dead-zone
remap resolves the amount). These catch the edge case where the policy outputs
action_type=Buy (or action_type=Sell) but amount_pct=0. Like the illegal penalties, these discourage
semantically empty trades.

#### `--turnover-penalty 0.05` and `--turnover-penalty-steps-threshold 15`

Penalty for selling positions too quickly (1 step = 1 minute). The goal is to punish the agent for executing many quick sells too impatiently. Sells afer an extended period of holding (> threshold) are not affected by this penalty.

---

### PPO Algorithm & Training

#### `--algorithm PPO`

Proximal Policy Optimisation with **action masking** (`MaskablePPO` from
`sb3_contrib`). Chosen over SAC because:

1. It natively supports discrete/multidiscrete action spaces.
2. It integrates cleanly with `ActionMasker` for hard constraint enforcement.
3. On-policy updates are safer for non-stationary financial data than SAC's
   replay buffer, which can mix stale and fresh market regimes.

#### `--gamma 0.990`

Discount factor controlling the effective planning horizon. At γ = 0.99 the
effective horizon is approximately 1/(1−γ) = 100 steps. At 1-min bars, this
corresponds to ~100 minutes of future reward lookahead — long enough to capture
meaningful price moves but short enough to remain numerically stable. A higher
gamma (0.999) would extend the horizon to ~1,000 steps but risks credit
assignment problems with sparse rewards. *The Optuna best-params found
γ = 0.961 (≈ 26-step horizon) — the current 0.99 is deliberately longer to
allow the bot to learn multi-step position management.*

#### `--learning-rate 0.00002`

Conservative learning rate (2e-5). The low rate reflects the noisy,
non-stationary nature of financial time-series: large gradient steps risk
catastrophic forgetting of previously learned patterns. The Optuna best-params
suggested 1.5e-5; the current value is slightly higher for marginally faster
convergence while remaining conservative relative to the SB3 PPO default (3e-4).

#### `--clip-range 0.23`

PPO's trust-region clipping parameter ε. At 0.23 this is close to the Optuna
best-params value (0.227), allowing moderately large policy updates per
iteration while staying within the PPO stability bound. A typical ε of 0.2 is
the SB3 default; 0.23 gives slightly more aggressive updates per rollout.

#### `--batch-size 64`

Minibatch size for each gradient update within a PPO epoch. The rollout buffer
holds `n_steps × n_envs = 256 × 8 = 2,048` transitions; these are divided into
`2048 / 64 = 32` minibatches per epoch. A smaller batch (64 vs. default 128)
introduces more gradient noise, which can act as a regulariser in high-dimensional
financial observation spaces. Confirmed by Optuna.

#### `--n-steps 256`

Number of environment steps collected per environment before a PPO update.
Together with `n_envs = 8`, each update uses 2,048 total transitions. Shorter
rollouts (256 vs. default 512) mean more frequent policy updates, improving
responsiveness to non-stationary data. Confirmed by Optuna as optimal.

#### `--n-envs 8`

Number of parallel environment instances collecting experience simultaneously.
8 envs balance CPU utilisation (each env is computationally lightweight but
observation calculation is non-trivial) against the overhead of Python
process synchronisation with `DummyVecEnv`.

#### `--ent-coef-initial 0.045` and `--ent-coef-final 0.001`

Entropy coefficient schedule, decayed linearly over the
training run via `EntropyDecayCallback`. High initial entropy encourages broad
exploration of the action space early in training; the low final value forces
convergence to a near-deterministic policy. The Optuna best-params
suggested a static ent_coef ≈ 0.047 throughout the entire experiment, but we will decay it over time.

---

### Environment & Data

#### `--fee-rate 0.001`

Realistic MEXC spot taker fee (0.1%). Applied to both buy and sell legs
of every trade. Fees are also explicitly penalised in the reward
(`fee_penalty` component) to make their cost visible to the agent.

#### `--budget-initial 100.0`

Starting cash in normalised USD units. The nominal value does not affect
learning outcomes (all rewards and observations are fractional/percentage-based),
but provides an intuitive absolute PnL reference in logs and dashboards.

#### `--window-size 240`

The observation includes the last 240 one-minute bars (= 4 hours) of relative
price changes per asset. Additionally, the observation includes fixed multi-scale
windows (30 × 1-min, 24 × 5-min, 24 × 60-min bars). The base window of 240
was confirmed by Optuna as optimal. Increasing it further grows the observation
vector linearly and is unlikely to add useful signal at this resolution.

#### `--n-rows 400000`

Number of rows loaded from the Parquet file. At 1-min resolution, 400,000 rows
≈ 277 days of multi-asset data, split 80/20 into train/test. Chosen as the
maximum that fits comfortably in RAM while providing sufficient regime diversity
(bull, bear, ranging) for generalisation.

#### `--parquet-path "binance_spot_1m_last4y_single_htf.parquet"`

The enhanced dataset that includes pre-computed **higher-time-frame (HTF)**
features: 15-minute slope, 1-hour slope, and 24-hour regime classification per
asset. These are included as static observation features to give the policy
structural context beyond raw 1-min bars without requiring the policy to learn
multi-scale aggregation from scratch.

---

### Cross-validation & Evaluation

#### `--cv-folds 1`

A single train/test split (80%/20%) is used. Walk-forward cross-validation with
multiple folds (`--cv-folds 3`) would produce a more robust estimate of
out-of-sample performance but triples training time. CV-folds = 1 is the fast
iteration default; increase to 3+ once the architecture stabilises.

#### `--timesteps 1500000`

Total training steps. 1.5M was chosen as the minimum required for the policy to
learn meaningful patterns from the 400k-row dataset; preliminary runs with fewer
steps showed insufficient convergence.

#### `--skip-multi-seed-eval`

Skips the post-training 5-seed robustness sweep to keep total wall-clock time
reasonable (~46 min for the last run). Re-enable for final candidate models
before deployment decisions.

#### `--checkpoint` and `--max-checkpoints 5`

Best-model checkpointing during training, scored by Calmar ratio. At most 5
checkpoints are retained. The Calmar ratio is preferred over raw PnL because
it jointly penalises low returns and high drawdowns.

---
