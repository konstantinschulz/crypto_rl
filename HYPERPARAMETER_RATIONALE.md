# Hyperparameter Rationale — `run_large.sh`

> **Purpose:** This document explains the *reasoning* behind each hyperparameter
> used in `scripts/run_large.sh`. It is intended as living documentation and as
> structured LLM input for diagnosing bot behaviour and identifying improvement
> opportunities. Update this file whenever you change a value in the launch
> script.

---

## 1. Hyperparameter Rationale

### Action Space

#### Target weights: `--max-asset-allocation 0.20`, `--weight-step 0.05`

The action is `MultiDiscrete([K] * num_assets)`: one target-weight bin per
asset, with bins `{0, 5, 10, 15, 20%}` of portfolio value
(`K = max_asset_allocation / weight_step + 1`). `--max-asset-allocation` must
be a multiple of `--weight-step`. If the targets sum to more than 100%, they
are scaled down proportionally, so "fully invested" is reachable without
picking exact bins. The rest is cash.

Each step the env rebalances toward the target at the last closed price.
Sells execute first so their proceeds fund the buys. If cash still runs short
(targets summing to 100% leave no room for fees), all buys are scaled down
together. Fees follow the same model as the baselines.

Every action is valid, so there is no action mask and plain `PPO` is used.
This replaced `MultiDiscrete([3, num_assets, 101])` (Hold/Buy/Sell × asset ×
size) with `MaskablePPO`, and with it the dead zone (`--action-dead-zone`),
the single-step cash cap (`--max-single-step-allocation`) and the
volatility-scaled cap (`--target-volatility`, which never bound). In that old
space, selling unheld assets silently became HOLD, the size dimension had no
effect on buys, and buys spent a fixed share of the remaining cash, giving
long tails of micro-buys (ROADMAP P1-3). Doing nothing is now "repeat the
current target". The observation already contains the current weights.

At 20% per asset at least 5 assets are needed to be fully invested, so the
bot cannot go all-in on one position.

#### `--rebalance-band 0.02`

No-trade band. An asset is traded only if its current weight is at least 2
percentage points off its target, so price drift does not cause a stream of
small, fee-paying rebalancing trades. A target of 0 always closes the whole
position, even a position smaller than the band. With a band larger than
`--weight-step`, the smallest bins cannot be reached from 0, so keep it below
the step.

#### `--decision-interval 15`

The agent decides every 15 one-minute bars (ROADMAP P2-1). Each env step
rebalances once at the last closed price, then holds the book for 15 bars;
the last step of an episode can be shorter. The observation is unchanged: it
is still built from 1-minute bars up to the last closed one, so the agent
sees the same fine-grained features but acts less often.

At 1-minute decisions the median absolute bar move was about 1/5 of the
round-trip fee, so the consequences of a single decision were mostly noise
and fees, and credit had to be assigned over hundreds of steps. At 15
minutes each decision moves the book over a span where price moves are
larger relative to fees, the discount horizon covers about a day in 100
steps, and each PPO step covers 15× more data.

Inside the interval, PV, the drawdown term and the benchmark are still
tracked bar by bar. The step reward is the sum of the 15 per-bar rewards,
which equals the reward of 15 one-minute steps holding the same book
(`tests/test_decision_interval.py`). `portfolio_values` keeps one mark per
minute.

Units: `--timesteps`, `--n-steps`, `--eval-freq`, `--batch-size` and
`--gamma` count decisions. `episode_length` (config only), `--eval-length`,
`--window-size` and `--turnover-penalty-steps-threshold` stay in one-minute
bars, so the episodes and eval windows cover the same time as before (a
7-day episode is 672 decisions). Calmar annualizes by elapsed minutes.
`--decision-interval 1` restores 1-minute decisions.

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
  one-minute bars of opening. Fees already penalize turnover through PV.

#### Evidence: reward ablation (2026-10-20)

Run with the old Hold/Buy/Sell action space, before P1-3.

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
always 0 because of a bookkeeping bug. The target-weight action space has no
illegal or empty actions to penalize.

---

### PPO Algorithm & Training

#### `--algorithm PPO`

Proximal Policy Optimisation (`PPO` from `stable_baselines3`). Chosen over
SAC because:

1. It natively supports discrete/multidiscrete action spaces.
2. On-policy updates are safer for non-stationary financial data than SAC's
   replay buffer, which can mix stale and fresh market regimes.

#### `--gamma 0.99`

Discount factor per **decision** step. The effective horizon is about
1/(1−γ) = 100 steps, which at `--decision-interval 15` is 1,500 minutes,
about 25 hours of lookahead: long enough to value holding a position through
the intraday moves that have to pay for its round-trip fee. At 1-minute
decisions the same γ would only look ~100 minutes ahead, and the old
γ = 0.992 (Optuna, 1-minute steps, with the leaky observation and linear
critic) only ~125 minutes. If you change `--decision-interval`, scale γ so
that `interval / (1 − γ)` stays near a day, e.g. γ ≈ 0.96 at 60 minutes.

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

#### Evidence: rule-based reference strategies (2026-10-21)

`scripts/reference_strategies.py` (ROADMAP P2-2) runs simple rules through the
same env as the agent, on the 28-day validation slice of the reward ablation
(2025-12-07 → 2026-01-04, 9 assets). `momentum_Xh` holds each asset with a
positive X-hour return in a fixed 10% sleeve. `basket_trend_Xh` holds the
equal-weight basket while it is above its X-hour moving average. Over the same
bars, the equal-weight basket returned +4.7% and BTC +3.3%. Return, with the
excess over equal weight in brackets:

| Strategy | 15-min decisions, fee 0.1% | Same, fee 0 | 60-min decisions, fee 0.1% | Fills (15 min) |
| --- | --- | --- | --- | --- |
| `momentum_1h` | −40.8% | +6.3% (+1.4 pp) | −23.5% | 5835 |
| `momentum_24h` | −19.2% | −9.6% (−14.5 pp) | −18.6% | 1127 |
| `basket_trend_1h` | −48.6% | +9.1% (+4.2 pp) | −20.6% | 6768 |
| `basket_trend_24h` | −17.2% | −6.5% (−11.4 pp) | −16.8% | 1090 |

- Before fees, the 1-hour signals beat equal weight by 1–4 pp over 28 days.
  The 24-hour signals lose to it even before fees: the window was choppy.
- With the 0.1% fee, every variant loses 17–49%. That gross edge needs
  thousands of fills, which cost far more than the edge earns. Deciding hourly
  halves the fees but still loses.
- So on this window, simple trend rules have no edge after fees at these
  horizons. The agent has to find something these rules miss, or trade
  much less. One 28-day window is a single draw, so check further folds
  (P2-3) before drawing firm conclusions.

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

#### `--cv-folds 3`

Walk-forward cross-validation with 3 expanding-window folds (ROADMAP P2-3). The
data is cut into 4 equal chunks and fold `i` tests on chunk `i + 1`, so the
three test folds tile the last three quarters of the data without overlap. Each
fold trains a fresh model. Its validation slice sits directly before its test
slice, and its training slice is everything before that. Each fold is scored
on its **whole** test fold (and on its whole validation slice in Optuna
trials). With 18M rows (~3.8 years) a test fold is about 347 days.

One fold gave a single window: one 7- or 14-day stretch with a handful of trades
is close to random. The run now reports, per fold, the excess log return over
the equal-weight basket, max drawdown, turnover and Calmar, and then the mean ±
sample std [min, max] of each across folds. Read the spread before the mean: a
positive mean whose minimum is far below zero is not an edge.

Costs: training time triples. `--test-fraction` is ignored with more than one
fold. `--val-fraction` is a fraction of the *whole* dataset, so with 3 folds
and `--val-fraction 0.1` the first fold trains on only about 60% of its chunk
(~208 days at 18M rows). Lower `--val-fraction` if the first fold's training
slice is too short. `--cv-folds 1` restores the single train/val/test split
(last `--test-fraction` of the data as test) for quick iteration.

#### `--objective-drawdown-coef 0.5`

λ in the model-selection score, computed per fold on an evaluation episode:

`score = log(PV_final / EW_final) − λ · maxDD`

`EW_final` is the equal-weight buy-and-hold basket over the same bars with the
same fees (the `equal_weight` baseline). `PV_final` is the bot's liquidation
value. maxDD is measured on the bar-by-bar mark-to-market PV. The same score:

- ranks checkpoints during training (on the fast validation slice),
- is reported, EMA-smoothed, to Optuna's pruner,
- is the Optuna objective: its mean over the folds' full validation slices.

It replaces the Calmar variant (`get_calmar_from_portfolio_series`), which is
discontinuous at zero return (gains are multiplied by the raw return, losses
are not) and annualizes short windows (×52 for 7 days). The new score is
continuous and monotonic: higher excess return is always better, a deeper
drawdown always worse, and it is not annualized. Calmar is still logged for
reference.

λ = 0.5 is a judgment call, not a tuned value. The trade-off it sets: holding the
basket (excess 0) through a 40% drawdown scores −0.20, while holding cash scores
−log(basket growth), so cash wins when the basket's log return is below half its
max drawdown. λ = 0 selects on excess return alone. Changing λ changes the
Optuna objective, so do it before a study, not during one.

#### `--test-fraction 0.1`

The fraction of the dataset that is used for evaluation. For large training runs, it is reduced from 0.2 to 0.1. Given a dataset size of about 1M rows, this still amounts to 100k rows for evaluation, which corresponds to about 70 days of 1-minute OHLCV data. Smaller runs (like in Optuna) still use the higher default value instead.

#### `--val-fraction 0.1`

The fraction of the dataset that is used for validation. Each fold is split into train / validation / test, in that chronological order. The validation slice is carved from the end of what would otherwise be the training slice, directly before the test slice, so the test slices do not depend on this value. Validation drives everything that selects or tunes a model: the intermediate evaluations during training, checkpoint ranking, Optuna pruning, and the Optuna objective (the mean over folds of the full-length validation score of the best checkpoint; see `--objective-drawdown-coef`). The test slice is only used for the final report (runs without an Optuna trial), once, over the whole test fold, with the best-on-validation checkpoint loaded together with the observation statistics it was trained with. Optuna trials never build or read the test slice. Because the checkpoint is chosen for its validation score, that score is optimistically biased; the test score is the honest number, and a large gap between the two in the final report is a sign of overfitting to the validation slice.

#### `--timesteps 4500000`

Total training steps. 4.5M was chosen as the minimum required for the policy to
learn meaningful patterns from the 18M-row dataset; preliminary runs with fewer
steps showed insufficient convergence. That was at 1-minute decisions. With
`--decision-interval 15` a step covers 15 bars, so 4.5M steps cover about 67M
bars, many passes over the ~1.6M training bars per asset. Fewer steps may be
enough, and watch the validation curve for overfitting.

#### `--skip-multi-seed-eval`

Skips the post-training 5-seed robustness sweep to keep total wall-clock time
reasonable (~46 min for the last run). Re-enable for final candidate models
before deployment decisions.

#### `--checkpoint` and `--max-checkpoints 5`

Best-model checkpointing during training, ranked by the validation score
(`--objective-drawdown-coef`): excess log return over the equal-weight basket
minus λ·maxDD. At most 5 checkpoints are retained. The score is preferred over
raw PnL because it penalises drawdowns, and over Calmar because it is
continuous at zero return and not annualized.

---
