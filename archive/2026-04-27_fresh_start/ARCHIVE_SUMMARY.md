# Archive Summary — experiments (2026-04-27 snapshot)

This document summarizes the archived experiments, scripts and the main insights recorded in the archive folder on 2026-04-27.

Location: `archive/2026-04-27_fresh_start/`

Contents (high level)
- `scripts/` — historical experiment runners and automated sweep scripts (e.g. `run_experiment9.py`, `run_heavy_training.py`, `run_manual_tuning_exp*.py`, `run_optuna_sweep.py`).
- `logs/` — run logs and parsed summaries (e.g. `manual_log*.txt`, `parsed_logs.txt`, `streamlit_smoke.log`).
- `docs/` — long-form experiment notes, results, and plans (`EXPERIMENT_LOG_v2.md`, `ALL_RESULTS.md`, `FUTURE_DIRECTIONS.md`, etc.).
- `states/` — historical dashboard state snapshots.
- `tensorboard/` and `artifacts/` — archived training artifacts and tensorboard exports.

Key scripts (examples)
- `run_manual_tuning_exp1.py` .. `run_manual_tuning_exp7.py`: handcrafted experiments testing reward shaping, invalid-action handling, and model hyperparameters.
- `run_heavy_training.py`: scaffolding for large-scale training with many timesteps and parallel envs.
- `run_optuna_sweep.py` / `optuna_test.py`: parameter sweeps (resource-heavy).

Main findings (condensed from `EXPERIMENT_LOG_v2.md` and `ALL_RESULTS.md`)
1. No-trade collapse and invalid-action degeneracy
   - Recurrent policies (LSTM) frequently converged to doing nothing (0 trades) on evaluation slices.
   - Root cause: the agent learned to choose invalid or degenerate actions (e.g. "sell 0%") that effectively avoid losses.

2. Action validity enforcement is necessary
   - Implementing action remapping (e.g. remap invalid sell → buy) restored trading but often caused extreme overtrading (one trade per step).
   - Penalize-only strategies or remapping to hold frequently caused collapse back to 0 trades.

3. Reward shaping has trade-offs
   - Strong inactivity penalties force the agent to trade, but can produce pathological policies that overtrade and lose to fees.
   - Shaped per-trade bonuses can be exploited to maximize trade count rather than trade quality.

4. Structural fixes outperform blind shaping
   - Best long-term improvements require changes to the action parameterization (discrete-valid heads or constrained outputs) and/or explicit trade-frequency regularization.

5. Transaction cost and per-trade budget matter
   - Lowering transaction-cost assumptions and limiting per-trade budget produced the first small positive Test returns in one configuration, but relied on optimistic fee assumptions.

Representative successful/failed configurations
- Exp 6G (best to date): small per-trade budgets + very low transaction costs → slight positive Test return (+0.03%), but not robust to realistic fees.
- Exp 7 (equity_delta reward): removed shaping, produced minimal trades (4 trades) and preserved risk-aversion, but did not discover profitable predictive edges.

Recommended next steps (from archived conclusions)
1. Prioritize structural changes to action space:
   - Introduce discrete, validity-constrained action heads and/or explicit masking to prevent invalid actions instead of remapping.
2. Increase scale and diversity:
   - Train longer (2–5M timesteps), increase lookback days (100–365 days), and add more symbols to provide richer signal.
3. Use parallelized envs for efficiency:
   - Use SubprocVecEnv with multiple workers to speed up sample collection and stabilize training.
4. Logging & diagnostics:
   - Record invalid-action rates, per-step trade counts, and richer per-step finance metrics for every training run.
5. Conservative next experiment:
   - Small structural change (valid-action head) + moderate increase in days (30→90) and a modest train step increase (e.g., 200k) as a sanity test before a full heavy run.

Where to find full details
- Read `archive/2026-04-27_fresh_start/docs/EXPERIMENT_LOG_v2.md` — detailed experiment log and rationales.
- Read `archive/2026-04-27_fresh_start/docs/ALL_RESULTS.md` — raw run outputs and summaries.

If you want, I can extract specific run IDs and produce a CSV summary table (run_id, command, val_return, test_return, trades, win_rate) from the logs so you can sort and filter historic results programmatically.

