from __future__ import annotations  # avoid circular import issues with TYPE_CHECKING

from typing import TYPE_CHECKING

import numpy as np

from crypto_rl.data import get_valid_start_timestamps, read_window_from_timestamps
from crypto_rl.env.data_utils import pivot_ohlcv
from crypto_rl.env.feature_utils import precalculate_static_obs
from crypto_rl.env.logging_utils import flush_log_parquet, init_log

if TYPE_CHECKING:  # avoid circular import issues
    from crypto_rl.env.minimal_env import MinimalCryptoEnv


def reset_env(env: MinimalCryptoEnv, seed=None, options=None):
    """Reset the MinimalCryptoEnv to its initial state."""
    # Note: caller or helper handles super().reset(seed=seed) if needed.
    if not env.config.disable_logging and env.log_buffer:
        flush_log_parquet(env)
    max_valid_start = len(env.prices_arr) - env.config.episode_length - 1
    if env.is_fast_eval:  # FAST INTERMEDIATE EVALUATION MODE: Run a fixed 2-week slice of the test fold
        # Start at exactly 25% into the test fold so it's safely past the window_size
        start_idx = env.config.window_size + (max_valid_start // 4)
        env.current_step = start_idx
        env.max_steps = min(start_idx + env.config.eval_length, len(env.prices_arr) - 1)
    elif env.is_eval or max_valid_start <= env.config.window_size:
        # FINAL EVALUATION MODE: Run the entire multi-month test fold start to finish
        env.current_step = env.config.window_size
        env.max_steps = len(env.prices_arr) - 1
    else:
        # TRAINING MODE: Random drop-ins
        env.current_step = env.np_random.integers(
            env.config.window_size, max_valid_start
        )
        env.max_steps = env.current_step + env.config.episode_length
    env.episode_count += 1
    if env.config.parquet_path is not None:
        if env._cached_valid_open_times is None:
            (
                env._cached_valid_open_times,
                env._cached_symbols,
                env._cached_k,
            ) = get_valid_start_timestamps(env.config.parquet_path, n=env.config.n_rows)
        new_df = read_window_from_timestamps(
            env.config.parquet_path,
            env._cached_valid_open_times,
            env._cached_symbols,
            env._cached_k,
        )
        # Pivot the OHLCV data
        (
            prices_piv,
            open_piv,
            high_piv,
            low_piv,
            volume_piv,
            htf_slope_15m_piv,
            htf_slope_1h_piv,
            htf_regime_24h_piv,
        ) = pivot_ohlcv(new_df)
        df_to_piv = {
            "prices_df": prices_piv,
            "open_df": open_piv,
            "high_df": high_piv,
            "low_df": low_piv,
            "volume_df": volume_piv,
            "htf_slope_15m_df": htf_slope_15m_piv,
            "htf_slope_1h_df": htf_slope_1h_piv,
            "htf_regime_24h_df": htf_regime_24h_piv,
        }
        # Ensure a fixed asset universe by reindexing to match original env.asset_names
        for k, v in df_to_piv.items():
            setattr(
                env, k, v.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
            )
        tr = env.high_df - env.low_df
        atr = tr.rolling(window=14, min_periods=1).mean()
        env.norm_vol_arr = (
            (atr / env.prices_df)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1e-8)
            .values.astype(np.float32)
        )
        assert env.prices_df.shape[1] == env.num_assets, (
            f"Asset count mismatch! Expected {env.num_assets} coins, "
            f"but sampled window only contained {env.prices_df.shape[1]}."
        )
        precalculate_static_obs(env)
        # If we dynamically loaded a specific window from disk, override the step trackers to match this newly loaded micro-dataset.
        env.current_step = env.config.window_size
        env.max_steps = len(env.prices_arr) - 1
    if not env.config.disable_logging:
        init_log(env, run_id=env.run_id)

    env.cash = env.config.budget_initial
    env.holdings = np.zeros(env.num_assets, dtype=np.float32)
    env.portfolio_value = env.config.budget_initial
    env.fees_paid_total = 0.0
    env.trades_count = 0
    env.avg_entry_price = np.zeros(env.num_assets, dtype=np.float32)
    # --- NEW: Reset entry step tracker ---
    env.entry_step.fill(0)
    env.total_cost_basis = np.zeros(env.num_assets, dtype=np.float32)
    # Reset new counters
    env.winning_trades_count = 0
    env.total_closed_trades = 0
    env.peak_portfolio_value = env.config.budget_initial
    env.per_asset_realized_pnl.fill(0.0)
    env.per_asset_trades.fill(0)
    env.per_asset_wins.fill(0)
    env.per_asset_fees.fill(0.0)
    env.previous_drawdown = 0.0
    return env._get_obs(), {"fees_paid": 0.0}
