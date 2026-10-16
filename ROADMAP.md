# Roadmap

Last verified against the code: 2026-10-16. Reference run: `latest_results.log`
(`scripts/run_large.sh`, 4.5M steps, 1 fold, `eval_length 10080`).

## TL;DR

The reference run's result (−7.07% vs. a "buy-and-hold baseline" of +179.5%) tells us
nothing about whether the strategy works, for three reasons that are bugs, not tuning issues:

1. **Every evaluation runs on training data.** The "test" memmap cache resolves to the
   train cache files, so intermediate evals, checkpoint selection, the Optuna objective and
   the final test all use the training fold.
2. **The observation leaks the bar the agent is paid for.** All precomputed indicators and
   the portfolio features are read at row `t`, while the step's reward is the return from
   `prices[t-1]` to `prices[t]`. RSI alone correlates +0.30 with the rewarded return.
3. **The value network is linear.** `net_arch` uses the SAC key `qf` instead of `vf`, so
   SB3 builds the PPO critic with no hidden layers.

The agent's behavior in the run is easy to explain: it buys every asset within the first
few steps and holds them. It returned −7.07% over the eval window, while an equal-weight
basket returned −7.02% over the same window. It is an index tracker that pays fees.

Fix the P0 items below before you trust another run or start another Optuna study.

---

## 1. Verification of the previous roadmap

| # | Previous claim | Verdict | What the code actually does |
| --- | --- | --- | --- |
| 1 | Buy-and-hold baseline is wrong | **Confirmed, but for a different reason** | It is **BTC-only** over the **whole fold**, not an equal-weight basket ([evaluation_utils.py:19](crypto_rl/experiment/evaluation_utils.py#L19)). The 279.50 is exactly BTC's ×2.795 over the **training** fold (Apr 2022 → Sep 2025), because the test data is the train data (see P0-1). Over the real test fold (2025-09-15 → 2026-01-31), BTC returned −31.8% and the equal-weight basket −47.9%. |
| 1 | The reward benchmark is inflated | **Confirmed** | The per-step benchmark is `max(0, market_return)` ([step.py:251](crypto_rl/env/step.py#L251)). Summed over the eval week, Σmax(0, mₜ) = **+2.67** while the market itself returned −0.068. This reproduces the logged `market_alpha` of −3.36 (≈ 1.2 × (−0.07 − 2.67)). With weights ≤ 1, alpha is ≤ 0 on almost every step whatever the policy does. The reward reduces to `1.2·(portfolio return) − 1.2·max(0, mₜ)`, a large noise term that depends on the market, not the policy. The 1.2× "asymmetry" ends up as nearly a constant scale factor. Being invested is punished as much as holding cash. |
| 2 | 398 BUY / 7 SELL; most BUYs blocked by the cap but still counted | **Partly falsified** | 335 of the 398 counted BUYs actually executed (342 fills = 335 buys + 7 sells). They are *micro-buys*. Each buy spends `min(cash·amount, cash·max_single_step_allocation, room)`, so cash shrinks geometrically: 40% of the remainder each time. With 9 assets × 20% cap = 180%, the cap rarely binds before cash runs out. Only ~63 BUYs were no-ops. |
| 2 | "Trades: 342" vs. per-asset sum of 16 | **Explained** | Two different counters. `trades_count` counts fills (buys and sells). `per_asset_trades` counts closes. The 16 closes are 7 policy sells **+ 9 forced terminal liquidations**, one per asset. The 42.9% win rate counts only the 7 policy sells. |
| 2 | Sell masking is structurally broken | **Confirmed** | The mask only checks "holds *any* asset" ([action_processing.py:148](crypto_rl/env/action_processing.py#L148)), and the asset dimension is sampled independently. |
| 2 | Illegal-sell penalty (7e-6) is effectively zero | **Worse: it is exactly zero** | `apply_discrete_action` adds the illegal penalties to `env._step_penalty`, then overwrites that value with the local `step_penalty` ([action_processing.py:136](crypto_rl/env/action_processing.py#L136)). `step_env` reads `_step_penalty` *before* the action is applied ([step.py:130](crypto_rl/env/step.py#L130)), so it gets last step's value. Empty-amount penalties can't fire because the dead zone turns amount 0 into HOLD first. The log shows `rule_penalties: 0.0`. |
| 2 | Positions are force-liquidated at episode end | **Confirmed** | All 9 assets were liquidated at the end of the reference run. |
| 3 | Dead zone 0.85 vs. 0.45 | **Doc drift confirmed** | The doc header says 0.45 and the body says 0.85. The run used 0.45; `config.py` now defaults to 0.50. |
| 3 | The dead zone wastes the size dimension | **Confirmed, and worse** | With dead zone 0.45 and `max_single_step_allocation` 0.40, every executed BUY spends exactly 40% of cash, so **the size dimension has no effect on buys at all**. Sells can only be 45–100% of the position. |
| 4 | 1-minute decisions vs. 0.2% round-trip fees | **Holds (design judgment)** | Median absolute 1-minute return on the eval slice is 0.043%, about 1/5 of the round-trip fee. |
| 5 | Too many weak shaping terms | **Mostly confirmed** | Cumulative values in the log: hold_cost −0.008, profit_bonus −0.0007, loss_cut +0.0003, fee −0.002, turnover 0, rule 0. In contrast, `capital_preservation_bonus` is 1e-3 **per step**, which could become the second-largest term if the agent ever held cash in a "bear" regime. |
| 5 | Add `VecNormalize(norm_reward=True)` | **Falsified, already done** | The training env already normalizes observations and rewards ([fold_runner.py:204](crypto_rl/experiment/fold_runner.py#L204)). |
| 6 | Evaluation can't distinguish skill from luck | **Confirmed, and worse** | Beyond the 7-day window: the "test" set is training data (P0-1). Checkpoint selection, Optuna and the final report all use the **same** slice. With `cv_folds=1`, the final test runs on the short fast-eval slice even though the comment says it runs the full fold ([fold_runner.py:430](crypto_rl/experiment/fold_runner.py#L430); the condition is inverted). The real test fold is about 139 days, not "~70 days". |
| 7 | Config / doc drift | **Confirmed, and wider** | There are three disagreeing sources: `config.py` defaults (lr 1e-4, clip 0.11, dz 0.50, dd 0.10), `run_large.sh` (dd 0.13, turnover **0.026**), and the logged run (dd **0.16**). Optuna's best `turnover_penalty` was **0.2595**, so `run_large.sh` has a 10× transcription error. `optuna_search.py` now uses 256 units, 18M rows and 3 folds, but the "best params" came from the older setup. |
| 7 | "pi and qf" is SAC terminology | **Confirmed, and it's a real bug** | It is not just wording. `fold_runner.py` passes `{"pi": ..., "qf": ...}` ([fold_runner.py:309](crypto_rl/experiment/fold_runner.py#L309)). SB3 2.7.1 reads `net_arch.get("vf", [])`, so the critic is a single linear layer. |
| 8 | HTF lookahead from resampling | **Falsified as stated, but a worse lookahead exists** | No resampling happens. The "HTF" features are EMAs of 1-minute closes ([preprocess_htf_indicators.py](scripts/preprocess_htf_indicators.py)) and are causal. The real leak is a one-bar offset that affects **every** indicator (P0-2). |

---

## 2. Current problems in the bot architecture

### A. Data and evaluation pipeline (results are currently invalid)

- **Cache-path singleton.** `CachePaths` is a class-level singleton that is initialized only
  once ([data_utils.py:95](crypto_rl/env/data_utils.py#L95),
  [data_utils.py:183](crypto_rl/env/data_utils.py#L183)). Every later `is_cache_built` or
  `compute_static_obs_from_long_df` call with a different prefix still points at the first
  prefix's files. The result: the test fold *is* the train fold, the `_test_*.npy` files are
  never built (only train files exist in `data/memmap_cache/`), and with `cv_folds>1` **all
  folds train and test on fold 0's train data**.
- **Stale cache keys.** The split cache key is only `seed + n_rows`
  ([runner.py:79](crypto_rl/experiment/runner.py#L79)), and the memmap prefixes don't include
  `test_fraction` or `cv_folds`. Changing those settings silently reuses the old split. The
  cached split is still the `test_fraction=0.1` split from the large run.
- **Model selection on the test set.** The intermediate eval, checkpoint ranking, Optuna
  pruning and objective, and the final test all use the same fixed slice. There is no
  separate validation set.
- **Checkpoints are never used.** The final test runs the *last* model, not the best
  checkpoint.
- **Checkpoint trimming deletes other runs' files.** It globs `logs/**/*.zip`
  ([callbacks.py:423](crypto_rl/callbacks.py#L423)), so it deletes checkpoints from other
  runs and other Optuna workers, plus the resume chunks in `logs/chunks/`.
- **Eval cadence depends on `n_envs`.** `num_timesteps % eval_freq == 0`
  ([callbacks.py:325](crypto_rl/callbacks.py#L325)) is only true at multiples of
  `lcm(n_envs, eval_freq)`, because `num_timesteps` advances by `n_envs`. For Optuna
  (1.5M steps, `eval_freq` 150k): `n_envs=9` gives 3 evals instead of 10, `n_envs=7` gives 1,
  and `n_envs=11` gives **none**, so that trial gets no pruning and no checkpoints. The
  dashboard callback has the same issue.
- **The baseline uses a different window and universe.** It is BTC-only over the whole fold,
  while the bot is measured on a 7- or 14-day slice of 9 assets.
- **Inconsistent metrics.** The fold Calmar (−28.73) and the `eval_report` Calmar (−35.80)
  disagree. `eval_report` reads the lexicographically last `actions_eval_*.parquet`, which is
  usually an intermediate eval episode (`ep9` sorts after `ep10`), not the final test. Its
  action counter (608 BUY / 108 SELL) is from a different episode than the fold result
  (398 / 7). `traded_coins` adds the selected asset on every step, including HOLD
  ([logging_utils.py:93](crypto_rl/env/logging_utils.py#L93)), so it always reports all 9
  coins.
- **Wrong date strings.** Timestamps are in microseconds but get formatted as nanoseconds, so
  the dashboard shows "1970-01-21".
- **Useless multi-seed eval.** The env and the policy are both deterministic, so all 5 seeds
  produce identical results.
- **Odd Calmar objective.** For losses it is annualized return / maxDD. For gains it is that
  value × raw return. The scale is discontinuous at 0, and annualizing a 7-day window
  multiplies the return by 52.

### B. Observation (lookahead)

- **Indicator leak.** `build_observation` uses `precalc_static_obs[t]` and `norm_vol_arr[t]`
  with `t = current_step`. The next `step()` executes at `prices[t-1]` and is rewarded with
  `prices[t]`. Row `t` of RSI, momentum, VWAP deviation, MACD, intrabar volatility (bar `t`'s
  high/low) and the EMA slopes already contains `prices[t]`. Measured on BTC: corr(RSI[t],
  rewarded return) = **+0.30**, versus +0.004 with the genuinely next return.
- **Portfolio-feature leak.** Holdings % and unrealized PnL use `prices[t]`
  ([observation.py:104](crypto_rl/env/observation.py#L104)), which leaks the next close for
  every held asset.
- **Mislabeled regime features.** The BTC "24h" and "7d" regime features use lags of 24 and
  168 **one-minute bars** ([feature_utils.py:223](crypto_rl/env/feature_utils.py#L223)).
  This regime also drives `capital_preservation_bonus`.
- **Inert volatility cap.** The cap uses `env.asset_volatility`, which is only set on the
  temporary env used to build the cache, so the real env always falls back to the static
  cap. Even when it is set, a 1-minute std of about 0.1% against `target_volatility` 0.02
  means it never binds.

### C. Reward

- **The floored benchmark** (`max(0, mₜ)` per minute) adds noise worth about 2.7 units per
  week that the policy can't influence. Since `VecNormalize` scales rewards by return
  variance, this noise also shrinks the useful signal.
- **Fees are counted twice.** They are already in the portfolio return and are subtracted
  again as `fee_penalty` ([step.py:232](crypto_rl/env/step.py#L232)).
- **Lump-sum terminal reward.** `terminal_return` adds the whole-episode return once at the
  end, on top of the per-step returns that already paid it out. The episode end is also
  flagged `terminated` rather than `truncated`, so PPO bootstraps an arbitrary 7-day cut as a
  true end of the world.
- **Rule penalties are broken** (always 0, see the table above).
- **The remaining shaping terms are negligible or conflicting.** `loss_cut_bonus` rewards
  realizing losses while the drawdown term punishes them. `capital_preservation_bonus` is
  keyed to a mislabeled 24-minute "bear" regime.

### D. Action space

- **The mask can't express per-asset sells**, so SELL on an unheld asset silently becomes
  HOLD, with no penalty in practice.
- **The size dimension is inert.** For buys it has no effect (dead zone > single-step cap).
  For sells, 45 of 101 bins are reachable.
- **Buys are geometric.** "40% of remaining cash" produces long tails of micro-buys, and fees
  scale with the number of fills.
- **Forced liquidation** at episode end distorts trade statistics and the last-step reward.

### E. Learning setup

- **Critic** without hidden layers (`qf` instead of `vf`).
- **Effective entropy schedule.** The decay targets `timesteps·(1−test_fraction)`
  ([fold_runner.py:373](crypto_rl/experiment/fold_runner.py#L373)), so entropy reaches its
  floor at 80–90% of training for no clear reason. The final value, 4e-4, encourages collapse
  into "buy everything and hold".
- **1-minute decisions** with a ~125-step horizon (γ=0.992), against fees about 5× larger
  than the median bar move.
- **Untrustworthy hyperparameters.** Every "Optuna-confirmed" value was optimized against
  training data (P0-1), with a leaky observation (P0-2) and a linear critic (P0-3).

---

## 3. Improvement plan

Priority: **P0** = results are invalid until fixed; **P1** = main obstacles to learning
anything useful; **P2** = makes the problem tractable and the results credible;
**P3** = hygiene. Effort: S < ½ day, M ≈ 1–2 days, L ≈ several days.

### P0: Make results trustworthy

| ID | Change | Impact | Effort |
| --- | --- | --- | --- |
| P0-1 | **Fix `CachePaths`.** Replace the class singleton with a per-prefix value, e.g. `CachePaths.for_prefix(prefix)` returning an instance. Include `test_fraction`, `cv_folds`, a feature-version tag and the data's date range in both the split cache key and the memmap prefix. Delete `data/memmap_cache/` afterwards. Add an assertion that the train and test arrays have different first timestamps. | Critical: today no number in the logs is out-of-sample | S |
| P0-2 | **Remove the one-bar lookahead.** In `build_observation`, read static features, `norm_vol_arr` and portfolio prices at `t-1`, the last *closed* bar before execution at `prices[t-1]`. Alternatively, keep `t` and execute and reward from `prices[t]` to `prices[t+1]`. Add a unit test asserting that no observation feature correlates with the rewarded return beyond noise. | Critical: any edge the agent finds now is probably the leak | S |
| P0-3 | **Set `net_arch={"pi": [...], "vf": [...]}`**, and fix the rationale doc. | High: a linear critic badly limits advantage estimates | S |
| P0-4 | **Separate validation from test.** Split each fold into train / val / test. Use val for intermediate evals, checkpoint ranking and the Optuna objective. Touch test only in the final report, with the **best** checkpoint loaded. Fix the inverted `is_fast_eval` condition so the final test always covers the whole test fold. | High: currently the model is selected on the same data it is scored on | M |
| P0-5 | **Use baselines on the exact same window.** Report equal-weight B&H, BTC B&H and all-cash over the same steps the bot is scored on, all with the same fee model. Log the bot's excess return over each. | High: without this the headline number can't be interpreted | S |
| P0-6 | **Fix eval and checkpoint bookkeeping.** Trigger evals on `n_calls` instead of `num_timesteps`, the way SB3's `EvalCallback` does. Restrict checkpoint trimming to the run's own `checkpoint_dir`. Have `eval_report` use the final-test log explicitly. Count `traded_coins` only on fills. Format timestamps with `unit="us"`. | Medium: Optuna trials currently get 0–3 evals depending on `n_envs`, and runs delete each other's files | S |

### P1: Give the agent a learnable problem

| ID | Change | Impact | Effort |
| --- | --- | --- | --- |
| P1-1 | **Simplify the reward.** Use `r_t = log(PV_t / PV_{t-1})`, net of fees, which are already inside PV. Optionally subtract the **unfloored** equal-weight log return, or add a small drawdown or differential-Sharpe term. Remove `fee_penalty` (double count), `terminal_return`, `loss_cut_bonus`, `profit_bonus`, `hold_cost`, `capital_preservation_bonus` and the broken rule penalties. Keep `VecNormalize(norm_reward=True)`. Add a shaping term back only if an ablation shows it changes behavior. | High: removes about 2.7 units per week of policy-independent noise and nine weak hyperparameters | S–M |
| P1-2 | **Use truncation instead of termination at the episode boundary,** and drop forced liquidation in training. Value open positions at mark-to-market, or charge the exit fee in the final PV without counting it as a "trade". | Medium: correct bootstrapping and honest trade statistics | S |
| P1-3 | **Use a target-weight action space,** e.g. `MultiDiscrete([K]*n_assets)` with bins {0, 5, 10, 15, 20%}, or a softmax over assets plus cash. The env rebalances toward the target, ideally with a no-trade band (e.g. skip if \|Δw\| < 2%) to keep fees down. This removes the dead zone, `max_single_step_allocation`, the action mask, and the illegal/empty penalties. | High: selling becomes as easy as buying, size becomes meaningful, and there are no illegal actions | M |
| P1-4 | **Fix the regime feature lags** (1440 / 10080 bars, or rename them), and either remove the volatility-scaled cap or compute `asset_volatility` in the real env. | Low–medium: the features currently mean something other than their names | S |

### P2: Make the problem tractable and the evidence credible

| ID | Change | Impact | Effort |
| --- | --- | --- | --- |
| P2-1 | **Decide every 15–60 minutes** (action repeat or a decision-bar wrapper) while keeping the 1-minute features. Raise γ to ≈0.99 at 15-minute steps (about 25 hours of lookahead). | High: decision consequences become larger relative to fees and noise, credit assignment gets ~15–60× shorter, and 4.5M env steps go much further | M |
| P2-2 | **Add a rule-based reference strategy,** e.g. 1h/24h momentum or a trend filter on the equal-weight basket, with the same fees and windows. If it can't beat fees, the RL agent is unlikely to either. | High: tells you early whether there is any edge to find | S |
| P2-3 | **Use at least 3 walk-forward folds,** with the full test fold per fold. Report the mean ± spread of excess return, maxDD and turnover, not just Calmar. Use a monotonic objective for Optuna, e.g. mean val excess log return − λ·maxDD. | Medium–high: a 7-day, 7-trade window is close to random | M |
| P2-4 | **Re-tune PPO** after P0/P1. Start at lr 3e-4, clip 0.2, a constant ent_coef around 0.01 or a gentler decay, and only then rerun Optuna. Narrow the search to the few hyperparameters that remain after P1-1. | Medium | M |

### P3: Hygiene

| ID | Change | Impact | Effort |
| --- | --- | --- | --- |
| P3-1 | **Use a single source of truth for configs.** Keep named presets (YAML or dataclass presets) instead of `config.py` defaults plus long shell commands plus the doc. Generate `HYPERPARAMETER_RATIONALE.md` values from the preset, or reference them. Fix `turnover_penalty` 0.026 vs. 0.2595 and the dead-zone, LR, `n_envs` and row-count statements. | Medium: stops tuned values from silently drifting | S |
| P3-2 | **Add regression tests:** train ≠ test arrays; no leak (P0-2); accounting identity (Σ per-asset PnL == PV change, which currently only produces a debug log); and buy/sell fee math. | Medium: prevents these bugs from coming back | M |
| P3-3 | **Reset `action_type_counts` per episode,** drop the dead multi-seed eval or seed real stochasticity (random start offsets), and remove the unused `total_cost_basis` and `chunk_buffer_path`. | Low | S |

### Suggested order

1. **P0-1 → P0-2 → P0-3** (about a day). Then rerun the existing config once. This gives you
   the first honest out-of-sample number. Expect it to be worse than what you've seen so far.
2. **P0-4, P0-5, P0-6, P2-2.** You now have a trustworthy evaluation harness and a simple
   benchmark to beat.
3. **P1-1, P1-2, P1-3** together, since they touch the same `step_env` code. Then **P2-1**.
4. **P2-3** and only then a fresh Optuna study (**P2-4**), on a narrowed search space.
5. **P3** items alongside, especially P3-2 tests for each P0 fix.

Throw away all existing Optuna results and "confirmed" hyperparameters. They were optimized
against training data, with a leaky observation and a linear critic.
