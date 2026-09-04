import numpy as np

from crypto_rl.env.data_utils import pivot_ohlcv
from crypto_rl.env.feature_utils import precalculate_static_obs
from crypto_rl.env.logging_utils import flush_log_parquet, init_log


def reset_env(env, seed=None, options=None):
    """Reset the MinimalCryptoEnv to its initial state."""
    # Note: caller or helper handles super().reset(seed=seed) if needed.
    if not env.disable_logging and env.log_buffer:
        flush_log_parquet(env)

    env.episode_count += 1

    if env.parquet_path is not None:
        from crypto_rl.data import (
            get_valid_start_timestamps,
            read_window_from_timestamps,
        )

        if env._cached_valid_open_times is None:
            (
                env._cached_valid_open_times,
                env._cached_symbols,
                env._cached_k,
            ) = get_valid_start_timestamps(env.parquet_path, n=env.n_rows)

        new_df = read_window_from_timestamps(
            env.parquet_path,
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

        # Ensure a fixed asset universe by reindexing to match original env.asset_names
        env.prices_df = (
            prices_piv.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
        )
        env.open_df = (
            open_piv.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
        )
        env.high_df = (
            high_piv.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
        )
        env.low_df = (
            low_piv.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
        )
        env.volume_df = (
            volume_piv.reindex(columns=env.asset_names).ffill().bfill().fillna(0.0)
        )
        env.htf_slope_15m_df = (
            htf_slope_15m_piv.reindex(columns=env.asset_names)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        env.htf_slope_1h_df = (
            htf_slope_1h_piv.reindex(columns=env.asset_names)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        env.htf_regime_24h_df = (
            htf_regime_24h_piv.reindex(columns=env.asset_names)
            .ffill()
            .bfill()
            .fillna(0.0)
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

    if not env.disable_logging:
        init_log(env, run_id=env.run_id)

    env.current_step = env.window_size
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
