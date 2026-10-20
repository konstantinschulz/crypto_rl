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

#### `--action-dead-zone 0.45`

In the multidiscrete action space, dimension 2 encodes trade size as an
integer 0–100 (mapped to 0.0–1.0). Any value below `action_dead_zone`
(i.e., amount_pct < 0.85) is silently converted to a **Hold** before the
action reaches the environment.

#### `--max-asset-allocation 0.20`

Hard cap on the fraction of total portfolio value that can be held in any single
asset at any point in time. Enforced inside `apply_discrete_action` by comparing
the current exposure to the cap before executing a buy. At 20% this allows
up to 5 assets to hold equal-weight positions, providing a minimum level of
diversification while still allowing meaningful concentration bets. Combined
with 9 tradeable assets, the bot cannot "go all-in" on one position.

#### `--max-single-step-allocation 0.40`

Maximum fraction of available **cash** that can be spent in a single buy
action. At 0.40, it allows the bot to spend a lot of cash at every single step, which gives it the power to commit quickly and heavily once a certain level of confidence is reached.

---

### Reward Function

#### `--reward-type excess_return`

The agent is rewarded for its **return over the equal-weight market return**
(alpha), not raw PnL. This incentivises the bot to actually beat a passive
index rather than just riding a bull market. A 1.2× asymmetric multiplier
is applied to negative alpha to slightly penalise underperformance more than
outperformance is rewarded (see `minimal_env.py` step logic). Rewards are
clipped to [−1.0, +1.0] to prevent variance explosion.

#### `--profit-bonus 0.065`

A shaped reward bonus applied on every **sell** that closes a position with
positive realised PnL. The bonus is proportional to the trade return
(`profit_bonus × trade_return`), so large profitable trades are rewarded more
than small ones. At 0.065 this is a modest supplement to the main alpha signal;
it is intended to reduce the agent's tendency to hold losing positions
indefinitely.

#### `--drawdown-penalty-coef 0.13`

Coefficient applied to the **delta drawdown** each step. Only worsening
drawdown steps are penalised (i.e., when the portfolio moves further below its
peak); recoveries are not rewarded to avoid encouraging risk-seeking behaviour.
It was decreased to 0.13 to allow trades enough breathing room to develop, even through short-term price fluctuations.

#### `--hold-cost-rate 0.0000004`

A micro-penalty applied to any position whose unrealised PnL
is worse than −5%. At 1-min bars, 4e-7 × 1440 min/day ≈ 0.06% per day — a
gentle but persistent drag on deeply underwater positions. The previous default
(1e-4) was catastrophically high (~6%/hour) and caused the agent to immediately
liquidate all holdings, so the value was reduced dramatically.

#### `--hold-penalty-threshold -0.014`

Unrealised-PnL threshold that gates the hold cost above. A position is only
charged `--hold-cost-rate` once its unrealised loss exceeds `|threshold|`, i.e.
1.4% below its average entry price. The implementation takes `abs()` of the
value, so the sign is cosmetic. The previous default (−0.03) waited until a
3% loss before any drag applied; the tighter −1.4% starts charging for
underwater positions earlier, which pushes the policy to cut or rework losers
sooner. Confirmed by Optuna (search range −0.10 to −0.01).

#### `--capital-preservation-bonus 0.001`

A flat per-step reward bonus paid while the BTC macro regime is bearish
(`btc_bear` feature > 0.5) and cash makes up more than 75% of portfolio value.
It is not scaled by PnL or position size, so it is a constant incentive to sit
in cash during downtrends. At 0.001 it is roughly 10× the default (0.0001) and
large enough to matter against the per-step alpha reward. Confirmed by Optuna
(search range 0.0 to 0.001).

#### `--loss-cut-bonus 0.0045`

Reward paid when a sell closes a losing trade (`trade_return < 0`) after the
position has been held for more than 15 steps. The bonus is
`|trade_return| × loss_cut_bonus`, so it scales with the size of the loss
realised, and it is intentionally small so it never outweighs avoiding the
loss in the first place. The 15-step minimum stops the bonus from rewarding
immediate, impulsive exits. At 0.0045 it is about 4.5× the default (0.001).
Confirmed by Optuna (search range 0.0 to 0.005).

---

### Penalty Structure

> **Design note:** The environment does **not** implement per-asset-conditional
> action masks (i.e., masking out specific assets on a per-sell basis). The
> action mask only gates the top-level action type (Buy/Sell). Individual
> asset-specific illegal actions (e.g., selling an asset you don't hold) are
> handled via reward penalties and forced remapping to Hold. The penalties must
> therefore be large enough to train the policy away from these actions without
> conditional masking.

#### `--illegal-buy-penalty 0.000007` and `--illegal-sell-penalty 0.000007`

Penalty added to the step reward when the agent attempts to:

- **Illegal buy**: BUY when cash ≈ 0 (cannot afford anything)
- **Illegal sell**: SELL an asset with zero holdings (short-selling attempt)

At 0.000007 these are smaller than the default `RULE_PENALTY` constant (5e-4). The rule_penalties cumulative value was **0.0** in the last eval, confirming the policy has successfully learned to
avoid illegal actions — the penalties have done their job and could potentially
be reduced.

#### `--empty-buy-penalty 0.000007` and `--empty-sell-penalty 0.000007`

Penalty for selecting Buy or Sell with a zero `amount_pct` (after the dead-zone
remap resolves the amount). These catch the edge case where the policy outputs
action_type=Buy (or action_type=Sell) but amount_pct=0. Like the illegal penalties, these discourage
semantically empty trades.

#### `--turnover-penalty 0.026` and `--turnover-penalty-steps-threshold 21`

Penalty for selling positions too quickly (1 step = 1 minute). The goal is to punish the agent for executing many quick sells too impatiently. Sells afer an extended period of holding (> threshold) are not affected by this penalty. In the last evaluation, turnover penalties were 0.0, indicating that the model did not sell any assets overly quickly.

---

### PPO Algorithm & Training

#### `--algorithm PPO`

Proximal Policy Optimisation with **action masking** (`MaskablePPO` from
`sb3_contrib`). Chosen over SAC because:

1. It natively supports discrete/multidiscrete action spaces.
2. It integrates cleanly with `ActionMasker` for hard constraint enforcement.
3. On-policy updates are safer for non-stationary financial data than SAC's
   replay buffer, which can mix stale and fresh market regimes.

#### `--gamma 0.992`

Discount factor controlling the effective planning horizon. At γ = 0.99 the
effective horizon is approximately 1/(1−γ) = 100 steps. At 1-min bars, this
corresponds to ~100 minutes of future reward lookahead — long enough to capture
meaningful price moves but short enough to remain numerically stable. A higher
gamma (0.999) would extend the horizon to ~1,000 steps but risks credit
assignment problems with sparse rewards. *The Optuna best-params found
γ = 0.992 (≈ 125-step horizon).*

#### `--learning-rate 0.00002`

Conservative learning rate (1e-5). The low rate reflects the noisy,
non-stationary nature of financial time-series: large gradient steps risk
catastrophic forgetting of previously learned patterns. The Optuna best-params
suggested 2e-5; the current value is conservative relative to the SB3 PPO default (3e-4).

#### `--clip-range 0.32`

PPO's trust-region clipping parameter ε. At 0.32 this allows moderately large policy updates per
iteration while staying within the PPO stability bound. A typical ε of 0.2 is
the SB3 default; 0.32 gives slightly more aggressive updates per rollout.

#### `--batch-size 128`

Minibatch size for each gradient update within a PPO epoch. The rollout buffer
holds `n_steps × n_envs = 1024 × 8 = 8,192` transitions; these are divided into
`8,192 / 128 = 64` minibatches per epoch. A smaller batch would
introduce more gradient noise, which can act as a regulariser in high-dimensional
financial observation spaces. Confirmed by Optuna.

#### `--n-steps 1024`

Number of environment steps collected per environment before a PPO update.
Together with `n_envs = 8`, each update uses 8,192 total transitions. Shorter
rollouts (e.g., 256) would mean more frequent policy updates, improving
responsiveness to non-stationary data. Confirmed by Optuna as optimal.

#### `--n-envs 9`

Number of parallel environment instances collecting experience simultaneously.
9 envs balance CPU utilisation (each env is computationally lightweight but
observation calculation is non-trivial) against the overhead of Python
process synchronisation with `DummyVecEnv`.

#### `--net-arch-dim 256`

Width of the hidden layers in the policy (`pi`) and value (`vf`) networks of the PPO algorithm. Each network has two hidden layers of this width, with ReLU activations. It was increased from 128 to 256 to allow the model to represent the complexities of a large dataset (about 4 years of 1-minute OHLCV data) more adequately. A value of 512 or higher would probably be too much, as the model might use this capacity to memorize the specific features of the training dataset. Smaller networks in financial reinforcement learning act as a regularizer for noisy crypto data with their very low signal-to-noise ratio.

The value network is the critic that estimates state values for the advantage calculation. Before the `vf` key fix it was built from the SAC key `qf`, which SB3 ignores for PPO, so the critic was a single linear layer with no hidden layers. Results from before the fix, including all Optuna runs (which used 128 dimensions), were produced with that linear critic and are not comparable to results from the current architecture.

#### `--ent-coef-initial 0.047` and `--ent-coef-final 0.0004`

Entropy coefficient schedule, decayed linearly over the
training run via `EntropyDecayCallback`. High initial entropy encourages broad
exploration of the action space early in training; the low final value forces
convergence to a near-deterministic policy. Confirmed by Optuna.

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

#### `--n-rows 18000000`

Number of rows loaded from the Parquet file. At 1-min resolution, 18M rows
≈ 3.8 years of multi-asset data, split 90/10 into train/test. Chosen as the
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

A 1-fold train/test split is used. Walk-forward cross-validation with
multiple folds (`--cv-folds 3`) would produce a more robust estimate of
out-of-sample performance but triples training time. CV-folds = 1 is the fast
iteration default, and is suitable for final single large training runs (after Optuna optimization with multi-fold CV per trial).

#### `--test-fraction 0.1`

The fraction of the dataset that is used for evaluation. For large training runs, it is reduced from 0.2 to 0.1. Given a dataset size of about 1M rows, this still amounts to 100k rows for evaluation, which corresponds to about 70 days of 1-minute OHLCV data. Smaller runs (like in Optuna) still use the higher default value instead.

#### `--val-fraction 0.1`

The fraction of the dataset that is used for validation. Each fold is split into train / validation / test, in that chronological order. The validation slice is carved from the end of what would otherwise be the training slice, directly before the test slice, so the test slices do not depend on this value. Validation drives everything that selects or tunes a model: the intermediate evaluations during training, checkpoint ranking, Optuna pruning, and the Optuna objective (the full-length validation Calmar of the best checkpoint). The test slice is only used for the final report (runs without an Optuna trial), once, over the whole test fold, with the best-on-validation checkpoint loaded together with the observation statistics it was trained with. Optuna trials never build or read the test slice. Because the checkpoint is chosen for its validation score, that score is optimistically biased; the test Calmar is the honest number, and a large gap between the two in the final report is a sign of overfitting to the validation slice.

#### `--timesteps 4500000`

Total training steps. 4.5M was chosen as the minimum required for the policy to
learn meaningful patterns from the 18M-row dataset; preliminary runs with fewer
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
