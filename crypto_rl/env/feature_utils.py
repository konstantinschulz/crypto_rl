import gc

import numpy as np
import pandas as pd

STATIC_PER_ASSET_DIM = 13  # 10 base statistical indicator features + 3 HTF indicators per asset in precalc_static_obs
MACRO_DIM = 5
# Map feature names to their exact indices in the macro block
MACRO_IDX = {
    "btc_mom_norm": 0,
    "btc_vol_norm": 1,
    "btc_bull": 2,
    "btc_ranging": 3,
    "btc_bear": 4,
}


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
    # Convert DataFrames to float32 numpy arrays (half the memory of float64)
    # ------------------------------------------------------------------
    env.prices_arr = env.prices_df.values.astype(np.float32)
    env.open_arr = env.open_df.values.astype(np.float32)
    env.high_arr = env.high_df.values.astype(np.float32)
    env.low_arr = env.low_df.values.astype(np.float32)
    env.volume_arr = env.volume_df.values.astype(np.float32)
    htf_indicators: dict[str, str] = {
        "htf_slope_15m_arr": "htf_slope_15m_df",
        "htf_slope_1h_arr": "htf_slope_1h_df",
        "htf_regime_24h_arr": "htf_regime_24h_df",
    }
    for indicator_name, indicator_df in htf_indicators.items():
        new_value = (
            getattr(env, indicator_df).values.astype(np.float32)
            if getattr(env, indicator_df) is not None
            else np.zeros_like(env.prices_arr)
        )
        setattr(env, indicator_name, new_value)
    # Keep column names before freeing the DataFrame
    env.asset_cols = list(env.prices_df.columns)

    # ------------------------------------------------------------------
    # Drop source DataFrames immediately - they are never needed again
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
    W = min(env.config.window_size, T)

    # Set static observation dimension metadata if missing
    env.macro_dim = getattr(env, "macro_dim", MACRO_DIM)
    env.macro_idx = MACRO_IDX
    env.static_per_asset_dim = STATIC_PER_ASSET_DIM
    env.static_dim = env.macro_dim + (env.static_per_asset_dim * N)

    if T <= W:
        raise ValueError(
            f"Dataset slice too short! `prices_df` has {T} timesteps, "
            f"but `window_size` is {W}. Increase `n_rows`."
        )

    # ------------------------------------------------------------------
    # All intermediate computations in numpy float32 to avoid pandas overhead.
    #
    # MEMORY STRATEGY: each intermediate array is deleted as soon as it has
    # been folded into a column block, so that at most a handful of large
    # arrays coexist at any point.  The final precalc_static_obs is assembled
    # via a single np.concatenate rather than a row-by-row Python loop (which
    # kept all 39+ (T, N) arrays alive simultaneously for the full T iters).
    # ------------------------------------------------------------------
    prices = env.prices_arr          # (T, N) float32 - owned by env, not freed here
    volume = env.volume_arr          # (T, N) float32
    high   = env.high_arr            # (T, N) float32
    low    = env.low_arr             # (T, N) float32
    htf_slope_15m  = env.htf_slope_15m_arr   # (T, N)
    htf_slope_1h   = env.htf_slope_1h_arr    # (T, N)
    htf_regime_24h = env.htf_regime_24h_arr  # (T, N)

    # Column blocks accumulate here; each has shape (T+1, k).
    # The extra row at [T] is a copy of [T-1] (boundary padding).
    col_blocks: list[np.ndarray] = []

    def _pad(arr: np.ndarray) -> np.ndarray:
        """Append a copy of the last row so shape becomes (T+1, ...)."""
        return np.vstack([arr, arr[-1:]])

    # --- returns: pct_change -------------------------------------------
    safe_prev = np.where(prices[:-1] > 0, prices[:-1], np.float32(1e-8))
    returns = np.empty_like(prices)
    returns[0] = 0.0
    returns[1:] = (prices[1:] - prices[:-1]) / safe_prev
    del safe_prev

    # --- rolling volatility (std of returns over W bars) ---------------
    vol_norm_arr = _rolling_std(returns, W)
    env.asset_volatility = vol_norm_arr
    del returns
    gc.collect()

    # --- momentum: prices[t]/prices[t-W] - 1 --------------------------
    momentum = np.zeros_like(prices)
    for t in range(W, T):
        ref = prices[t - W]
        safe_ref = np.where(ref > 0, ref, np.nan)
        momentum[t] = prices[t] / safe_ref - 1.0
    momentum = np.nan_to_num(momentum, nan=0.0)

    # --- RSI (EWM, alpha=1/14) -----------------------------------------
    delta = np.diff(prices, axis=0, prepend=prices[:1])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    del delta
    alpha = 1.0 / 14.0
    avg_gain = _ewm(gain, alpha);  del gain
    avg_loss = _ewm(loss, alpha);  del loss
    gc.collect()
    rs = avg_gain / (avg_loss + 1e-8)
    del avg_gain, avg_loss
    rsi_raw = (rs / (1.0 + rs)) * 2.0 - 1.0
    del rs

    # --- MACD: (mean_3 - mean_W) / std_W --------------------------------
    mean_W, std_W = _rolling_mean_std(prices, W)
    mean_3, _ = _rolling_mean_std(prices, min(3, T))
    macd_raw = (mean_3 - mean_W) / (std_W + 1e-8)
    del mean_3, std_W, mean_W

    # --- Base Volume normalisation (Window W) --------------------------
    vol_mean, _ = _rolling_mean_std(volume, W)
    safe_vol_mean = np.where(vol_mean > 0, vol_mean, np.nan);  del vol_mean
    vol_norm = np.nan_to_num((volume / safe_vol_mean) - 1.0, nan=0.0).astype(np.float32)
    del safe_vol_mean

    # --- 24h Relative Volume (RVOL) ------------------------------------
    W_24h = min(1440, T)
    vol_mean_24h, _ = _rolling_mean_std(volume, W_24h)
    safe_vol_mean_24h = np.where(vol_mean_24h > 0, vol_mean_24h, np.nan)
    del vol_mean_24h
    rvol_24h = np.nan_to_num(volume / safe_vol_mean_24h, nan=1.0).astype(np.float32)
    del safe_vol_mean_24h

    # --- Intrabar volatility -------------------------------------------
    safe_prices = np.where(prices > 0, prices, np.nan)
    intrabar_vol = np.nan_to_num((high - low) / safe_prices, nan=0.0).astype(np.float32)
    del safe_prices, high, low

    # --- Base VWAP deviation (Window W) --------------------------------
    pv = prices * volume
    pv_sum  = _rolling_sum(pv, W)
    vol_sum = _rolling_sum(volume, W)
    safe_vol_sum = np.where(vol_sum > 0, vol_sum, np.nan);  del vol_sum
    vwap = pv_sum / safe_vol_sum;  del pv_sum, safe_vol_sum
    safe_vwap = np.where(vwap > 0, vwap, np.nan);  del vwap
    vwap_dev = np.nan_to_num((prices / safe_vwap) - 1.0, nan=0.0).astype(np.float32)
    del safe_vwap

    # --- 24h VWAP deviation --------------------------------------------
    pv_sum_24h  = _rolling_sum(pv, W_24h)
    vol_sum_24h = _rolling_sum(volume, W_24h)
    del pv, volume   # no longer needed
    safe_vol_sum_24h = np.where(vol_sum_24h > 0, vol_sum_24h, np.nan)
    del vol_sum_24h
    vwap_24h = pv_sum_24h / safe_vol_sum_24h;  del pv_sum_24h, safe_vol_sum_24h
    safe_vwap_24h = np.where(vwap_24h > 0, vwap_24h, np.nan);  del vwap_24h
    vwap_dev_24h = np.nan_to_num(
        (prices / safe_vwap_24h) - 1.0, nan=0.0
    ).astype(np.float32)
    del safe_vwap_24h
    gc.collect()

    # --- BTC macro features & correlations ----------------------------
    btc_idx = env.asset_cols.index("BTCUSDT") if "BTCUSDT" in env.asset_cols else 0
    btc_prices = prices[:, btc_idx]
    btc_mom    = momentum[:, btc_idx]
    btc_vol    = vol_norm_arr[:, btc_idx]

    # --- 60m BTC Correlation -----------------------------------------
    W_60m = min(60, T)
    btc_corr_60m = _rolling_corr(prices, btc_prices[:, None], W_60m)

    # 24h and 7d % change for BTC regime
    btc_mom_24h = _pct_change_lag(btc_prices, min(24, T))
    btc_mom_7d  = _pct_change_lag(btc_prices, min(168, T))
    del btc_prices
    bull_mask    = (btc_mom_24h > 0.005)  & (btc_mom_7d > 0.005)
    bear_mask    = (btc_mom_24h < -0.005) & (btc_mom_7d < -0.005)
    ranging_mask = ~(bull_mask | bear_mask)
    del btc_mom_24h, btc_mom_7d

    btc_regime_1hot = np.zeros((T, 3), dtype=np.float32)
    btc_regime_1hot[bull_mask, 0] = 1.0
    btc_regime_1hot[ranging_mask, 1] = 1.0
    btc_regime_1hot[bear_mask, 2] = 1.0
    del bull_mask, bear_mask, ranging_mask

    # Relative momentum vs BTC
    rel_mom = momentum - btc_mom[:, None]

    # Rolling z-scores
    mom_norm     = _rolling_zscore(momentum, window=min(100, T), min_periods=10);  del momentum
    rel_mom_norm = _rolling_zscore(rel_mom,  window=min(100, T), min_periods=10);  del rel_mom

    btc_mom_norm = _rolling_zscore(btc_mom[:, None], window=min(100, T), min_periods=10)[:, 0]
    del btc_mom
    btc_vol_norm = _rolling_zscore(btc_vol[:, None], window=min(100, T), min_periods=10)[:, 0]
    del btc_vol
    gc.collect()

    # ------------------------------------------------------------------
    # Assemble precalc_static_obs: shape (T+1, static_dim)
    #
    # VECTORIZED: build column blocks and concatenate once instead of
    # filling row-by-row in a Python loop (which kept all intermediate
    # arrays alive for the full T iterations).
    # Each array is appended to col_blocks then immediately deleted so
    # the peak footprint is bounded to a few blocks at a time.
    # ------------------------------------------------------------------

    # Macro block (T+1, MACRO_DIM=5)
    macro_block = np.zeros((T + 1, MACRO_DIM), dtype=np.float32)
    macro_block[:T, 0]    = btc_mom_norm
    macro_block[:T, 1]    = btc_vol_norm
    macro_block[:T, 2:5]  = btc_regime_1hot
    macro_block[T]        = macro_block[T - 1]
    col_blocks.append(macro_block)
    del btc_mom_norm, btc_vol_norm, btc_regime_1hot, macro_block

    # Per-asset indicator blocks (each (T+1, N)).
    # Array is padded, appended, then deleted immediately.
    for arr in [
        vol_norm,
        intrabar_vol,
        vwap_dev,
        mom_norm,
        rsi_raw,
        macd_raw,
        rel_mom_norm,
        rvol_24h,
        vwap_dev_24h,
        btc_corr_60m,
        htf_slope_15m,
        htf_slope_1h,
        htf_regime_24h,
    ]:
        col_blocks.append(_pad(arr))
        del arr

    # Explicitly unbind the local names so CPython refcount drops to zero
    del (vol_norm, intrabar_vol, vwap_dev, mom_norm, rsi_raw, macd_raw,
         rel_mom_norm, rvol_24h, vwap_dev_24h, btc_corr_60m,
         htf_slope_15m, htf_slope_1h, htf_regime_24h)
    gc.collect()

    # Single concatenation
    env.precalc_static_obs = np.concatenate(col_blocks, axis=1)
    del col_blocks
    gc.collect()

    # Clamp any residual NaN / Inf
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
    W = min(W, T)
    out = np.zeros((T, N), dtype=np.float32)
    cs = np.cumsum(arr, axis=0, dtype=np.float32)
    out[W - 1 :] = cs[W - 1 :]
    if W > 1:
        out[W:] -= cs[:-W]
    return out


def _rolling_mean_std(arr: np.ndarray, W: int):
    """Returns (mean, std) arrays of shape (T, N) float32, window W."""
    T, N = arr.shape
    W = min(W, T)
    arr32 = arr.astype(np.float32)
    cs = np.cumsum(arr32, axis=0)
    cs2 = np.cumsum(arr32**2, axis=0)

    mean = np.zeros((T, N), dtype=np.float32)
    std = np.zeros((T, N), dtype=np.float32)

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


def _rolling_corr(A: np.ndarray, B: np.ndarray, W: int) -> np.ndarray:
    """Rolling Pearson correlation matrix between two broadcastable arrays."""
    W = min(W, A.shape[0])
    mean_A, std_A = _rolling_mean_std(A, W)
    mean_B, std_B = _rolling_mean_std(B, W)
    mean_AB, _ = _rolling_mean_std(A * B, W)

    cov = mean_AB - (mean_A * mean_B)
    corr = cov / (std_A * std_B + 1e-8)
    # Clip to correct any extremely minor precision overflow past bounds [-1.0, 1.0]
    return np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0).astype(np.float32)


def _rolling_zscore(
    arr: np.ndarray, window: int = 100, min_periods: int = 10
) -> np.ndarray:
    """Row-wise rolling z-score with min_periods, result float32."""
    T, N = arr.shape
    if T == 0:
        return np.zeros((T, N), dtype=np.float32)

    window = min(window, T)
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
    T = len(arr)
    lag = min(lag, T)
    out = np.zeros(T, dtype=np.float32)
    prev = arr[:-lag]
    safe = np.where(prev > 0, prev, np.nan)
    out[lag:] = np.nan_to_num((arr[lag:] - prev) / safe, nan=0.0)
    return out
