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

#### `--reward-type excess_log_return`

The step reward is the portfolio log return minus the log return of an
equal-weight basket over the same bar:
`r_t = log(PV_t / PV_{t-1}) − log(mean_i(p_i,t / p_i,t-1))`. The benchmark is
unfloored, so it is negative in down markets. Fees are paid out of cash, so
they are already inside PV and counted exactly once. There is no lump-sum
terminal reward and no clipping.

`--reward-type log_return` drops the benchmark. Its per-step rewards telescope:
over an episode they sum to `log(PV_final / PV_initial)`, with open positions
at mark-to-market.

The episode boundary is a time limit, so the last step is flagged `truncated`
(not `terminated`) and PPO bootstraps the value of the open book. Positions are
not force-liquidated. The reported final PV (`final_portfolio_value`) charges
the exit fee on open positions, the same way the passive baselines are charged,
but no sell is executed and it is not counted as a trade or a closed trade.

Training rewards are scaled by `VecNormalize(norm_reward=True)`, so the absolute
size of the log return does not matter.

#### Optional shaping: `--drawdown-penalty-coef` and `--turnover-penalty`

Both default to 0 (off).

- `--drawdown-penalty-coef` multiplies the per-step **increase** in drawdown
  (recoveries are not rewarded).
- `--turnover-penalty` with `--turnover-penalty-steps-threshold` charges
  `turnover_penalty × closed weight` for sells within the threshold number of
  steps of opening. Fees already penalize turnover through PV.

#### Evidence: reward ablation (2026-10-20)

`scripts/reward_ablation.py`: 4 arms × 3 seeds × 600k PPO steps on 3.6M rows,
scored on the 28-day validation slice (2025-12-07 → 2026-01-04). The test
fold was not used. Over that window the equal-weight basket returned +4.7%.
Mean ± std over seeds:

| Reward | Return | Max DD | Mean exposure | Turnover (× PV) | Fees |
| --- | --- | --- | --- | --- | --- |
| `log_return` | −19.4 ± 7.1% | 0.29 | 0.83 | 262 | 22.7% |
| `log_return` + drawdown 0.5 | −26.8 ± 3.6% | 0.30 | 0.44 | 321 | 27.0% |
| `excess_log_return` | −11.6 ± 4.9% | 0.22 | 0.82 | 148 | 13.3% |
| `excess_log_return` + drawdown 0.5 | −10.6 ± 3.0% | 0.11 | 0.13 | 109 | 10.1% |

- `excess_log_return` roughly halved turnover and fees at the same exposure.
  That is why it is the default. The return gap overlaps across seeds.
- The drawdown term clearly changes behavior: exposure falls from about 0.83
  to 0.44 or 0.13, against a seed spread of about 0.03. It hurt with
  `log_return` and halved max drawdown with the excess reward. It stays
  opt-in until a longer run shows whether that trade-off is worth it.
- Every arm lost money to fees. The training runs were short and had not
  converged, so this shows which way each option pushes behavior, not how a
  trained agent performs.

#### Removed terms

`fee_penalty` (double-counted fees), `terminal_return` (re-paid the whole
episode return at the end), `profit_bonus`, `loss_cut_bonus`, `hold_cost`
(`--hold-cost-rate`, `--hold-penalty-threshold`), `capital_preservation_bonus`
and the illegal/empty BUY/SELL rule penalties are gone. The rule penalties were
always 0 because of a bookkeeping bug. Illegal and empty actions are still
remapped to HOLD, with no penalty.

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
of every trade. Fees reduce PV directly, so the log-return reward already
accounts for them.

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
