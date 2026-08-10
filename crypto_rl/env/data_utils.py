import pandas as pd
import numpy as np

def pivot_ohlcv(prices_df: pd.DataFrame):
    """Pivot OHLCV columns into separate DataFrames for each asset.
    Guarantees that columns always match the asset names.
    Returns a tuple: (close_df, open_df, high_df, low_df, volume_df).
    """
    def _pivot_and_align(val_col: str) -> pd.DataFrame:
        pivoted = prices_df.pivot(
            index="open_time", columns="symbol", values=val_col
        )
        # Align columns to existing asset order if already set on env later
        # Here we just forward-fill and back-fill missing values.
        return pivoted.ffill().bfill().fillna(0.0)

    close_df = _pivot_and_align("close")
    open_df = _pivot_and_align("open")
    high_df = _pivot_and_align("high")
    low_df = _pivot_and_align("low")
    volume_df = _pivot_and_align("volume")
    return close_df, open_df, high_df, low_df, volume_df
