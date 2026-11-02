# CLAUDE.md

Notes for coding agents working in this repo. `ROADMAP.md` tracks open problems,
and `HYPERPARAMETER_RATIONALE.md` explains config values.

## Commands

- Python: `.conda/bin/python` (the project's conda env).
- Tests: `.conda/bin/python -m pytest -q tests`
- Lint/format: `.conda/bin/python -m ruff check <files>` and `ruff format`.

## Pitfalls

### `MinimalCryptoEnv` ignores its arrays when `config.parquet_path` is set

If `config.parquet_path` is not `None`, `reset()` (`crypto_rl/env/reset.py`)
discards the `prices_arr` / `static_obs` / `norm_vol_arr` passed to the
constructor. It streams a random window from the **whole** parquet file and
recomputes the indicators. Two consequences:

- **Leakage:** training windows can come from the validation or test period.
- **Speed:** about 7× slower (≈11 s per reset with 9 envs).

There is no error or warning. Whenever you build an env from precomputed
slices (memmap caches, test fixtures, scripts), pass a config with
`parquet_path=None` (and `n_rows=0`), as `runner.py` does for `env_config`:

```python
env_cfg = dataclasses.replace(config, parquet_path=None, n_rows=0)
```

The default `RLConfig()` has `parquet_path` set, so a bare `RLConfig()` is
affected.

### `LoggerBase` methods raise `NotImplementedError`

`MinimalCryptoEnv` defaults to `LoggerBase()`, and `load_raw_data` /
`prepare_splits` call `logger.print_if_not_trial`. Outside `selective_logger`
(`crypto_rl/env/logging_utils.py`), pass a subclass that implements `debug`,
`info`, `log` and `print_if_not_trial`. Otherwise these calls crash, for
example the env's end-of-episode PnL-mismatch debug message.
`scripts/reward_ablation.py` has a minimal `PrintLogger`.

### Evaluation hygiene

Use the validation slice for experiments, ablations and model selection. The
test slice is only for the final report (ROADMAP P0-4).

On 2026-10-26 the test slices of folds 1–2 of the 18M-row, 3-fold layout
(2023-03-27 → 2025-02-18) were used for signal research
(`scripts/signal_walk_forward.py`). Only the fold-3 test slice
(2025-02-18 → 2026-01-31) was still an untouched holdout after that.

On 2026-10-28 that fold-3 slice was used once, for the frozen Kraken
low-turnover strategy (`scripts/kraken_backtest.py --include-holdout`, full
panel `data/coin_panel_15m_full.npz`). No untouched historical test data
remains. New ideas need fresh data (after 2026-01-31) or forward/paper
trading for an unbiased check.
