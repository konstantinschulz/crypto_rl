import pandas as pd
import numpy as np

def precalculate_static_obs(env) -> None:
    """Pre-calculate static observation matrix and related arrays for the environment.
    Mutates the given env instance, setting attributes:
    - prices_arr, open_arr, high_arr, low_arr, volume_arr
    - static_dim, macro_dim, static_per_asset_dim, obs_buf, unrealised_pnl_buf
    - precalc_static_obs
    """
    # Convert dataframes to numpy arrays
    env.prices_arr = env.prices_df.values.astype(np.float32)
    env.open_arr = env.open_df.values.astype(np.float32)
    env.high_arr = env.high_df.values.astype(np.float32)
    env.low_arr = env.low_df.values.astype(np.float32)
    env.volume_arr = env.volume_df.values.astype(np.float32)
    T = env.prices_df.shape[0]
    N = env.num_assets
    W = env.window_size
    if T <= W:
        raise ValueError(
            f"Dataset slice too short! `prices_df` has {T} timesteps, "
            f"but `window_size` is {W}. Increase `n_rows`."
        )
    # Base Returns & Rolling Volatility per coin
    returns_df = env.prices_df.pct_change()
    vol_df = returns_df.rolling(window=W, min_periods=W).std()
    # Rolling Momentum per coin
    momentum_df = (env.prices_df / env.prices_df.shift(W).replace(0.0, np.nan)) - 1.0
    # RSI [-1, 1]
    delta = env.prices_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs_df = avg_gain / (avg_loss + 1e-8)
    rsi_df = (rs_df / (1 + rs_df)) * 2.0 - 1.0
    # Normalized MACD
    rolling_window = env.prices_df.rolling(window=W, min_periods=W)
    mean_all = rolling_window.mean()
    std_all = rolling_window.std()
    mean_3 = env.prices_df.rolling(window=3, min_periods=3).mean()
    macd_df = (mean_3 - mean_all) / (std_all + 1e-8)
    # New OHLCV features
    vol_mean = (
        env.volume_df.rolling(window=W, min_periods=W).mean().replace(0.0, np.nan)
    )
    vol_norm = ((env.volume_df / vol_mean) - 1.0).fillna(0.0).values.astype(np.float32)
    # Intrabar volatility
    intrabar_vol = (
        ((env.high_df - env.low_df) / env.prices_df.replace(0.0, np.nan))
        .fillna(0.0)
        .values.astype(np.float32)
    )
    # VWAP deviation
    vwap_denom = (
        env.volume_df.rolling(window=W, min_periods=W).sum().replace(0.0, np.nan)
    )
    vwap = (env.prices_df * env.volume_df).rolling(window=W, min_periods=W).sum() / vwap_denom
    vwap_dev = ((env.prices_df / vwap.replace(0.0, np.nan)) - 1.0).fillna(0.0).values.astype(np.float32)
    # BTC Macro Features
    btc_symbol = (
        "BTCUSDT" if "BTCUSDT" in env.prices_df.columns else env.prices_df.columns[0]
    )
    btc_prices = env.prices_df[btc_symbol]
    btc_mom = momentum_df[btc_symbol]
    btc_vol = vol_df[btc_symbol]
    btc_mom_24h = btc_prices.pct_change(24)
    btc_mom_7d = btc_prices.pct_change(168)
    bull_mask = (btc_mom_24h > 0.005) & (btc_mom_7d > 0.005)
    bear_mask = (btc_mom_24h < -0.005) & (btc_mom_7d < -0.005)
    btc_regime_1hot = np.zeros((T, 3), dtype=np.float32)
    btc_regime_1hot[bull_mask.fillna(False).values, 0] = 1.0
    btc_regime_1hot[~(bull_mask | bear_mask).fillna(True).values, 1] = 1.0
    btc_regime_1hot[bear_mask.fillna(False).values, 2] = 1.0
    rel_mom_df = momentum_df.sub(btc_mom, axis=0)
    def rolling_zscore(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        r = df.rolling(window=window, min_periods=10)
        return (df - r.mean()) / (r.std() + 1e-8)
    mom_norm = rolling_zscore(momentum_df).fillna(0.0).values.astype(np.float32)
    rel_mom_norm = rolling_zscore(rel_mom_df).fillna(0.0).values.astype(np.float32)
    macd_raw = macd_df.fillna(0.0).values.astype(np.float32)
    rsi_raw = rsi_df.fillna(0.0).values.astype(np.float32)
    btc_mom_norm = rolling_zscore(btc_mom.to_frame())[btc_symbol].fillna(0.0).values.astype(np.float32)
    btc_vol_norm = rolling_zscore(btc_vol.to_frame())[btc_symbol].fillna(0.0).values.astype(np.float32)
    # Pack static features
    env.precalc_static_obs = np.zeros((T + 1, env.static_dim), dtype=np.float32)
    def safe_relative_change(window_arr, ref_price):
        denom = np.where(ref_price > 0, ref_price, 1e-8)
        res = (window_arr / denom) - 1.0
        res[:, ref_price <= 0] = 0.0
        return res.ravel()
    for t in range(W, T):
        idx = 0
        env.precalc_static_obs[t, idx] = btc_mom_norm[t]
        env.precalc_static_obs[t, idx + 1] = btc_vol_norm[t]
        env.precalc_static_obs[t, idx + 2 : idx + 5] = btc_regime_1hot[t]
        idx += env.macro_dim
        last_price = env.prices_arr[t - 1]
        # Base window
        window = env.prices_arr[t - W : t]
        env.precalc_static_obs[t, idx : idx + W * N] = safe_relative_change(window, last_price)
        idx += W * N
        # 1‑min 30‑bar
        if t >= 30:
            win_1min = env.prices_arr[t - 30 : t]
            norm_1min = safe_relative_change(win_1min, last_price)
        else:
            norm_1min = np.zeros(30 * N, dtype=np.float32)
        env.precalc_static_obs[t, idx : idx + 30 * N] = norm_1min
        idx += 30 * N
        # 5‑min 24‑bar
        if t >= 5 * 24:
            idxs_5 = np.arange(t - 5 * 24, t, 5)
            win_5min = env.prices_arr[idxs_5]
            norm_5min = safe_relative_change(win_5min, last_price)
        else:
            norm_5min = np.zeros(24 * N, dtype=np.float32)
        env.precalc_static_obs[t, idx : idx + 24 * N] = norm_5min
        idx += 24 * N
        # 60‑min 24‑bar
        if t >= 60 * 24:
            idxs_60 = np.arange(t - 60 * 24, t, 60)
            win_60min = env.prices_arr[idxs_60]
            norm_60min = safe_relative_change(win_60min, last_price)
        else:
            norm_60min = np.zeros(24 * N, dtype=np.float32)
        env.precalc_static_obs[t, idx : idx + 24 * N] = norm_60min
        idx += 24 * N
        # Indicators
        env.precalc_static_obs[t, idx : idx + N] = vol_norm[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = intrabar_vol[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = vwap_dev[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = mom_norm[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = rsi_raw[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = macd_raw[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = rel_mom_norm[t]
        idx += N
    env.precalc_static_obs[T] = env.precalc_static_obs[T - 1]
    env.precalc_static_obs = np.nan_to_num(
        env.precalc_static_obs, nan=0.0, posinf=0.0, neginf=0.0
    )
