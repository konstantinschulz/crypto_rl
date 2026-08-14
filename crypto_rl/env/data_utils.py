from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_rl.env.feature_utils import precalculate_static_obs


import gc
import numpy as np
import pandas as pd


def pivot_ohlcv(prices_df: pd.DataFrame):
    """
    Pivot OHLCV columns into separate DataFrames for each asset.
    Destroys the source DataFrame incrementally to prevent massive RAM spikes.
    """

    def _pivot_and_align(val_col: str) -> pd.DataFrame:
        pivoted = prices_df.pivot(index="open_time", columns="symbol", values=val_col)

        # Destroy the column in the original long_df immediately to prevent memory doubling
        if val_col in prices_df.columns:
            prices_df.drop(columns=[val_col], inplace=True)
            gc.collect()

        # Forward-fill / back-fill missing values, then cast to float32
        return pivoted.ffill().bfill().fillna(0.0).astype(np.float32)

    close_df = _pivot_and_align("close")
    open_df = _pivot_and_align("open")
    high_df = _pivot_and_align("high")
    low_df = _pivot_and_align("low")
    volume_df = _pivot_and_align("volume")

    # Destroy the remaining skeleton of the source DataFrame (symbol, open_time)
    prices_df.drop(columns=prices_df.columns, inplace=True)
    gc.collect()

    return close_df, open_df, high_df, low_df, volume_df


def compute_static_obs_from_long_df(
    long_df: pd.DataFrame, window_size: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot long-format OHLCV DataFrame and pre-calculate static observations.

    Returns:
        tuple: (prices_arr, static_obs, asset_names)

    Memory notes
    ------------
    - The pivoted DataFrames are float32 (not float64) thanks to pivot_ohlcv.
    - precalculate_static_obs converts them to numpy *and* sets the DataFrame
      slots to None, so they are freed before this function returns.
    """
    prices_df, open_df, high_df, low_df, volume_df = pivot_ohlcv(long_df)
    asset_names = prices_df.columns.tolist()
    num_assets = len(asset_names)

    temp_env = SimpleNamespace(
        prices_df=prices_df,
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        window_size=window_size,
        num_assets=num_assets,
    )
    precalculate_static_obs(temp_env)
    # After this call, temp_env.prices_df / open_df / … are all None (freed
    # inside precalculate_static_obs).  Only numpy arrays remain.
    return temp_env.prices_arr, temp_env.precalc_static_obs, asset_names
