import gc

import numpy as np
import pandas as pd

STATIC_PER_ASSET_DIM = (
    10  # 7 base statistical indicator features + 3 HTF indicators per asset in precalc_static_obs
)
MACRO_DIM = 5


def add_volatility_normalized_features(
    df: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    # Calculate True Range (TR)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # 14-period ATR
    atr = tr.rolling(window=window).mean()

    # Relative ATR (ATR as percentage of price)
    df["norm_volatility"] = atr / df["close"]

    # Volatility-normalized log returns
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["norm_return"] = log_ret / (df["norm_volatility"] + 1e-8)

    return df


def precalculate_static_obs(env) -> None:
    """Pre-calculate static observation matrix and related arrays for the environment.

    Mutates the given env instance, setting attributes:
    - prices_arr, open_arr, high_arr, low_arr, volume_arr
    - htf_slope_15m_arr, htf_slope_1h_arr, htf_regime_24h_arr
    - static_dim, macro_dim, static_per_asset_dim, obs_buf, unrealised_pnl_buf
    - precalc_static_obs

    After computation the source DataFrames (prices_df, open_df, etc.) are
    deleted from the env to free the memory they occupied; only the derived
    numpy arrays are retained.
    """
    # ------------------------------------------------------------------
    # Convert DataFrames → float32 numpy arrays (half the memory of float64)
    # ------------------------------------------------------------------
    env.prices_arr = env.prices_df.values.astype(np.float32)
    env.open_arr = env.open_df.values.astype(np.float32)
    env.high_arr = env.high_df.values.astype(np.float32)
    env.low_arr = env.low_df.values.astype(np.float32)
    env.volume_arr = env.volume_df.values.astype(np.float32)

    # HTF indicators
    env.htf_slope_15m_arr = (
        env.htf_slope_15m_df.values.astype(np.float32)
        if getattr(env, "htf_slope_15m_df", None) is not None
        else np.zeros_like(env.prices_arr)
    )
    env.htf_slope_1h_arr = (
        env.htf_slope_1h_df.values.astype(np.float32)
        if getattr(env, "htf_slope_1h_df", None) is not None
        else np.zeros_like(env.prices_arr)
    )
    env.htf_regime_24h_arr = (
        env.htf_regime_24h_df.values.astype(np.float32)
        if getattr(env, "htf_regime_24h_df", None) is not None
        else np.zeros_like(env.prices_arr)
    )

    # Keep column names before freeing the DataFrame
    env.asset_cols = list(env.prices_df.columns)

    # ------------------------------------------------------------------
    # Drop source DataFrames immediately – they are never needed again
    # ------------------------------------------------------------------
    env.prices_df = None
    env.open_df = None
    env.high_df = None
    env.low_df = None
    env.volume_df = None
    env.htf_slope_15m_df = None
    env.htf_slope_1h_df = None
    env.htf_regime_24h_df = None
    gc.collect()  # Force memory release before allocating precalc_static_obs

    T, N = env.prices_arr.shape
    W = env.window_size

    # Set static observation dimension metadata if missing
    env.macro_dim = getattr(env, "macro_dim", MACRO_DIM)
    env.static_per_asset_dim = STATIC_PER_ASSET_DIM
    env.static_dim = env.macro_dim + (env.static_per_asset_dim * N)
    if T <= W:
        raise ValueError(
            f"Dataset slice too short! `prices_df` has {T} timesteps, "
            f"but `window_size` is {W}. Increase `n_rows`."
        )

    # ------------------------------------------------------------------
    # All intermediate computations in numpy float32 to avoid pandas overhead
    # ------------------------------------------------------------------
    prices = env.prices_arr  # (T, N) float32
    volume = env.volume_arr  # (T, N) float32
    high = env.high_arr  # (T, N) float32
    low = env.low_arr  # (T, N) float32
    htf_slope_15m = env.htf_slope_15m_arr  # (T, N) float32
    htf_slope_1h = env.htf_slope_1h_arr  # (T, N) float32
    htf_regime_24h = env.htf_regime_24h_arr  # (T, N) float32

    # --- returns: pct_change -------------------------------------------
    # returns[t] = (prices[t] - prices[t-1]) / prices[t-1]
    safe_prev = np.where(prices[:-1] > 0, prices[:-1], 1e-8)
    returns = np.empty_like(prices)  # (T, N)
    returns[0] = 0.0
    returns[1:] = (prices[1:] - prices[:-1]) / safe_prev  # (T-1, N)

    # --- rolling volatility (std of returns over W bars) ---------------
    # Uses a vectorised O(T*N) cumsum approach
    vol_norm_arr = _rolling_std(returns, W)  # (T, N) float32
    env.asset_volatility = vol_norm_arr
    # --- momentum: prices[t]/prices[t-W] - 1 --------------------------
    momentum = np.zeros_like(prices)  # (T, N)
    for t in range(W, T):
        ref = prices[t - W]
        safe_ref = np.where(ref > 0, ref, np.nan)
        momentum[t] = prices[t] / safe_ref - 1.0
    momentum = np.nan_to_num(momentum, nan=0.0)

    # --- RSI (EWM, alpha=1/14) -----------------------------------------
    delta = np.diff(prices, axis=0, prepend=prices[:1])  # (T, N)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    alpha = 1.0 / 14.0
    avg_gain = _ewm(gain, alpha)  # (T, N)
    avg_loss = _ewm(loss, alpha)  # (T, N)
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_raw = (rs / (1.0 + rs)) * 2.0 - 1.0  # (T, N), float32

    # --- MACD: (mean_3 - mean_W) / std_W --------------------------------
    mean_W, std_W = _rolling_mean_std(prices, W)  # (T, N)
    mean_3, _ = _rolling_mean_std(prices, 3)  # (T, N)
    macd_raw = (mean_3 - mean_W) / (std_W + 1e-8)  # (T, N)

    # --- Volume normalisation -----------------------------------------
    vol_mean, _ = _rolling_mean_std(volume, W)
    safe_vol_mean = np.where(vol_mean > 0, vol_mean, np.nan)
    vol_norm = np.nan_to_num((volume / safe_vol_mean) - 1.0, nan=0.0).astype(np.float32)

    # --- Intrabar volatility ------------------------------------------
    safe_prices = np.where(prices > 0, prices, np.nan)
    intrabar_vol = np.nan_to_num((high - low) / safe_prices, nan=0.0).astype(np.float32)

    # --- VWAP deviation -----------------------------------------------
    pv = prices * volume  # price × volume
    pv_sum = _rolling_sum(pv, W)  # (T, N)
    vol_sum = _rolling_sum(volume, W)  # (T, N)
    safe_vol_sum = np.where(vol_sum > 0, vol_sum, np.nan)
    vwap = pv_sum / safe_vol_sum  # (T, N)
    safe_vwap = np.where(vwap > 0, vwap, np.nan)
    vwap_dev = np.nan_to_num((prices / safe_vwap) - 1.0, nan=0.0).astype(np.float32)

    # --- BTC macro features -------------------------------------------
    btc_idx = env.asset_cols.index("BTCUSDT") if "BTCUSDT" in env.asset_cols else 0
    btc_prices = prices[:, btc_idx]  # (T,)
    btc_mom = momentum[:, btc_idx]  # (T,)
    btc_vol = vol_norm_arr[:, btc_idx]  # (T,)

    # 24h and 7d % change for BTC regime
    btc_mom_24h = _pct_change_lag(btc_prices, 24)  # (T,)
    btc_mom_7d = _pct_change_lag(btc_prices, 168)  # (T,)
    bull_mask = (btc_mom_24h > 0.005) & (btc_mom_7d > 0.005)
    bear_mask = (btc_mom_24h < -0.005) & (btc_mom_7d < -0.005)
    ranging_mask = ~(bull_mask | bear_mask)

    btc_regime_1hot = np.zeros((T, 3), dtype=np.float32)
    btc_regime_1hot[bull_mask, 0] = 1.0
    btc_regime_1hot[ranging_mask, 1] = 1.0
    btc_regime_1hot[bear_mask, 2] = 1.0

    # Relative momentum vs BTC
    rel_mom = momentum - btc_mom[:, None]  # (T, N)

    # Rolling z-score (window=100, min 10 periods)
    mom_norm = _rolling_zscore(momentum, window=100, min_periods=10)  # (T, N)
    rel_mom_norm = _rolling_zscore(rel_mom, window=100, min_periods=10)  # (T, N)

    btc_mom_2d = btc_mom[:, None]  # shape (T,1) for zscore func
    btc_vol_2d = btc_vol[:, None]
    btc_mom_norm = _rolling_zscore(btc_mom_2d, window=100, min_periods=10)[:, 0]  # (T,)
    btc_vol_norm = _rolling_zscore(btc_vol_2d, window=100, min_periods=10)[:, 0]  # (T,)

    # ------------------------------------------------------------------
    # Assemble precalc_static_obs: shape (T+1, static_dim)
    # ------------------------------------------------------------------
    env.precalc_static_obs = np.zeros((T + 1, env.static_dim), dtype=np.float32)

    def safe_relative_change(
        window_arr: np.ndarray, ref_price: np.ndarray
    ) -> np.ndarray:
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

        # Indicators only
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
        env.precalc_static_obs[t, idx : idx + N] = htf_slope_15m[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = htf_slope_1h[t]
        idx += N
        env.precalc_static_obs[t, idx : idx + N] = htf_regime_24h[t]
        idx += N

    env.precalc_static_obs[T] = env.precalc_static_obs[T - 1]
    np.nan_to_num(env.precalc_static_obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Vectorised helper functions (all float32)
# ---------------------------------------------------------------------------


def _ewm(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential weighted mean along axis 0 (adjust=False)."""
    T, N = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = arr[0]
    one_minus = np.float32(1.0 - alpha)
    a = np.float32(alpha)
    for t in range(1, T):
        out[t] = one_minus * out[t - 1] + a * arr[t]
    return out


def _rolling_sum(arr: np.ndarray, W: int) -> np.ndarray:
    """Sliding window sum along axis 0, result in float32."""
    T, N = arr.shape
    out = np.zeros((T, N), dtype=np.float32)
    cs = np.cumsum(arr, axis=0, dtype=np.float32)
    out[W - 1 :] = cs[W - 1 :]
    out[W:] -= cs[:-W]
    return out


def _rolling_mean_std(arr: np.ndarray, W: int):
    """Returns (mean, std) arrays of shape (T, N) float32, window W."""
    T, N = arr.shape
    arr32 = arr.astype(np.float32)
    cs = np.cumsum(arr32, axis=0)
    cs2 = np.cumsum(arr32**2, axis=0)

    mean = np.zeros((T, N), dtype=np.float32)
    std = np.zeros((T, N), dtype=np.float32)

    # Full windows: rows W-1 .. T-1
    s = cs[W - 1 :]
    s2 = cs2[W - 1 :]
    if W > 1:
        s = s - np.vstack([np.zeros((1, N), dtype=np.float32), cs[:-W]])
        s2 = s2 - np.vstack([np.zeros((1, N), dtype=np.float32), cs2[:-W]])
    mean[W - 1 :] = s / W
    var = np.clip(s2 / W - (s / W) ** 2, 0, None)
    std[W - 1 :] = np.sqrt(var)
    return mean, std


def _rolling_std(arr: np.ndarray, W: int) -> np.ndarray:
    """Rolling std of shape (T, N) float32, window W."""
    _, std = _rolling_mean_std(arr, W)
    return std


def _rolling_zscore(
    arr: np.ndarray, window: int = 100, min_periods: int = 10
) -> np.ndarray:
    """Row-wise rolling z-score with min_periods, result float32."""
    T, N = arr.shape
    if T == 0:
        return np.zeros((T, N), dtype=np.float32)

    arr32 = arr.astype(np.float32)
    cs = np.cumsum(arr32, axis=0)
    cs2 = np.cumsum(arr32**2, axis=0)

    t_idx = np.arange(T)
    w_start = np.maximum(0, t_idx - window + 1)
    w_len = (t_idx - w_start + 1)[:, None]

    cs_prev = np.zeros((T, N), dtype=np.float32)
    cs2_prev = np.zeros((T, N), dtype=np.float32)
    if window < T:
        cs_prev[window:] = cs[:-window]
        cs2_prev[window:] = cs2[:-window]

    s = cs - cs_prev
    s2 = cs2 - cs2_prev

    mu = s / w_len
    var = np.clip(s2 / w_len - mu**2, 0, None)
    sig = np.sqrt(var)

    valid = (w_len >= min_periods) & (sig > 1e-8)
    safe_sig = np.where(sig > 1e-8, sig, 1.0)
    return np.where(valid, (arr32 - mu) / safe_sig, 0.0).astype(np.float32)


def _pct_change_lag(arr: np.ndarray, lag: int) -> np.ndarray:
    """Percentage change against `lag` steps back, shape (T,) float32."""
    out = np.zeros(len(arr), dtype=np.float32)
    prev = arr[:-lag]
    safe = np.where(prev > 0, prev, np.nan)
    out[lag:] = np.nan_to_num((arr[lag:] - prev) / safe, nan=0.0)
    return out
