import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_rl.env.feature_utils import precalculate_static_obs


def pivot_ohlcv(prices_df: pd.DataFrame):
    """
    Pivot OHLCV and HTF columns into separate DataFrames for each asset.
    Destroys the source DataFrame incrementally to prevent massive RAM spikes.
    """

    def _pivot_and_align(val_col: str) -> pd.DataFrame:
        pivoted = prices_df.pivot(index="open_time", columns="symbol", values=val_col)

        # Destroy the column in the original long_df immediately to prevent memory doubling
        if val_col in prices_df.columns:
            del prices_df[val_col]
            gc.collect()

        # Forward-fill / back-fill missing values, then cast to float32
        return pivoted.ffill().bfill().fillna(0.0).astype(np.float32)

    close_df = _pivot_and_align("close")
    open_df = _pivot_and_align("open")
    high_df = _pivot_and_align("high")
    low_df = _pivot_and_align("low")
    volume_df = _pivot_and_align("volume")

    # Handle HTF indicator columns if present in long_df, otherwise fallback compute
    if "htf_slope_15m" in prices_df.columns:
        htf_slope_15m_df = _pivot_and_align("htf_slope_15m")
    else:
        ema_15 = close_df.ewm(span=15, adjust=False).mean()
        htf_slope_15m_df = ((ema_15 - ema_15.shift(15)) / (close_df + 1e-8)).fillna(0.0).astype(np.float32)

    if "htf_slope_1h" in prices_df.columns:
        htf_slope_1h_df = _pivot_and_align("htf_slope_1h")
    else:
        ema_60 = close_df.ewm(span=60, adjust=False).mean()
        htf_slope_1h_df = ((ema_60 - ema_60.shift(60)) / (close_df + 1e-8)).fillna(0.0).astype(np.float32)

    if "htf_regime_24h" in prices_df.columns:
        htf_regime_24h_df = _pivot_and_align("htf_regime_24h")
    else:
        ema_1440 = close_df.ewm(span=1440, adjust=False).mean()
        htf_regime_24h_df = ((close_df - ema_1440) / (ema_1440 + 1e-8)).fillna(0.0).astype(np.float32)

    # Destroy the remaining skeleton of the source DataFrame (symbol, open_time)
    del prices_df
    gc.collect()

    return (
        close_df,
        open_df,
        high_df,
        low_df,
        volume_df,
        htf_slope_15m_df,
        htf_slope_1h_df,
        htf_regime_24h_df,
    )


def compute_static_obs_from_long_df(
    long_df: pd.DataFrame, window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Pivot long-format OHLCV DataFrame and pre-calculate static observations.

    Returns:
        tuple: (prices_arr, static_obs, norm_vol_arr, asset_names)

    Memory notes
    ------------
    - The pivoted DataFrames are float32 (not float64) thanks to pivot_ohlcv.
    - precalculate_static_obs converts them to numpy *and* sets the DataFrame
      slots to None, so they are freed before this function returns.
    """
    (
        prices_df,
        open_df,
        high_df,
        low_df,
        volume_df,
        htf_slope_15m_df,
        htf_slope_1h_df,
        htf_regime_24h_df,
    ) = pivot_ohlcv(long_df)
    asset_names = prices_df.columns.tolist()
    num_assets = len(asset_names)
    # --- NEW: Calculate 14-period ATR Normalized Volatility ---
    # Fast proxy for True Range using High - Low
    tr = high_df - low_df
    atr = tr.rolling(window=14, min_periods=1).mean()
    norm_vol_df = atr / prices_df
    # Replace NaNs/Infs and fill to prevent division by zero
    norm_vol_arr = (
        norm_vol_df.replace([np.inf, -np.inf], np.nan)
        .fillna(1e-8)
        .values.astype(np.float32)
    )
    # ----------------------------------------------------------
    temp_env = SimpleNamespace(
        prices_df=prices_df,
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        htf_slope_15m_df=htf_slope_15m_df,
        htf_slope_1h_df=htf_slope_1h_df,
        htf_regime_24h_df=htf_regime_24h_df,
        window_size=window_size,
        num_assets=num_assets,
    )
    precalculate_static_obs(temp_env)
    # After this call, temp_env.prices_df / open_df / … are all None (freed
    # inside precalculate_static_obs).  Only numpy arrays remain.
    return temp_env.prices_arr, temp_env.precalc_static_obs, norm_vol_arr, asset_names
