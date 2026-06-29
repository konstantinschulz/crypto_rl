"""
crypto_rl.data
==============
Data loading utilities for the crypto trading environment.

The main entry point is :func:`read_last_n`, which loads a random
contiguous time window of OHLCV data for a fixed set of symbols from a
Parquet file.  It prefers a row-group-aware PyArrow reader for memory
efficiency but falls back to pandas if PyArrow is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Try to use pyarrow for efficient, row-group-aware parquet reads so we
# don't load the entire file into memory.  Fallback to pandas.read_parquet
# if pyarrow is not available.
try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq  # noqa: F401 – kept to mirror original guard
except Exception:
    pq = None  # type: ignore
    ds = None  # type: ignore

# The canonical list of symbols to trade, in priority order.
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def read_last_n(path: str, n: int = 10000) -> pd.DataFrame:
    """Load a random contiguous time window of price data.

    Parameters
    ----------
    path:
        Path to a Parquet file that contains at least the columns
        ``symbol``, ``open_time``, and ``close``.
    n:
        Approximate number of rows to return.  The window is sized so
        that each of the selected symbols contributes ``n // num_assets``
        rows.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ``[symbol, open_time, close]``,
        sorted by ``open_time`` then ``symbol``, with no NaN values.
    """
    cols = ["symbol", "open_time", "close"]

    if ds is None or pq is None:
        return _read_last_n_pandas(path, n, cols)
    return _read_last_n_pyarrow(path, n, cols)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pick_symbols(available: list[str]) -> list[str]:
    """Return the preferred symbols that are present in *available*."""
    symbols = [s for s in DEFAULT_SYMBOLS if s in available]
    if not symbols:
        symbols = available[:5]
    return symbols


def _random_window(times: np.ndarray, k: int) -> tuple:
    """Return (t_start, t_end) for a random window of length *k*."""
    if len(times) <= k:
        return times[0], times[-1]
    start_idx = np.random.randint(0, len(times) - k)
    return times[start_idx], times[start_idx + k - 1]


def _read_last_n_pandas(path: str, n: int, cols: list[str]) -> pd.DataFrame:
    """Fallback path used when PyArrow is unavailable."""
    df = pd.read_parquet(path, columns=cols).dropna()
    symbols = _pick_symbols(list(df["symbol"].unique()))
    k = max(1, n // len(symbols))

    df_sub = df[df["symbol"].isin(symbols)]
    anchor_times = np.sort(df_sub[df_sub["symbol"] == symbols[0]]["open_time"].unique())
    t_start, t_end = _random_window(anchor_times, k)

    mask = (df_sub["open_time"] >= t_start) & (df_sub["open_time"] <= t_end)
    return (
        df_sub[mask]
        .sort_values(by=["open_time", "symbol"])
        .reset_index(drop=True)
    )


def _read_last_n_pyarrow(path: str, n: int, cols: list[str]) -> pd.DataFrame:
    """Memory-efficient path using PyArrow row-group-aware reads."""
    dataset = ds.dataset(path, format="parquet")
    symbols = DEFAULT_SYMBOLS
    k = max(1, n // len(symbols))

    # Load open times of BTCUSDT to determine a random time window
    btc_filter = ds.field("symbol") == "BTCUSDT"
    btc_open_times = (
        dataset.to_table(columns=["open_time"], filter=btc_filter)
        .column("open_time")
        .to_numpy()
    )

    t_start, t_end = _random_window(btc_open_times, k)

    query_filter = (
        ds.field("symbol").isin(symbols)
        & (ds.field("open_time") >= t_start)
        & (ds.field("open_time") <= t_end)
    )
    df = dataset.to_table(columns=cols, filter=query_filter).to_pandas()
    return (
        df.dropna()
        .sort_values(by=["open_time", "symbol"])
        .reset_index(drop=True)
    )
