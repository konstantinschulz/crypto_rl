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

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "UNIUSDT",
]


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


def _get_symbol_start_times(dataset: ds.Dataset, symbols: list[str]) -> dict[str, int]:
    """Fast metadata scan to find the minimum (earliest) timestamp for each symbol,
    bypassing Arrow dictionary unification errors by using pandas per-symbol extraction.
    """
    start_times = {}

    # Scanning symbol by symbol avoids cross-chunk dictionary type mismatches
    for symbol in symbols:
        try:
            symbol_filter = ds.field("symbol") == symbol
            # Pull only open_time for this specific symbol
            t = dataset.to_table(columns=["open_time"], filter=symbol_filter)
            if t.num_rows > 0:
                # Use numpy min on the chunked array directly (very fast, zero-copy conversion)
                min_time = np.min(t.column("open_time").to_numpy())
                start_times[symbol] = int(min_time)
        except Exception as e:
            print(f"[Warning] Could not retrieve start time for {symbol}: {e}")

    return start_times


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
    return df_sub[mask].sort_values(by=["open_time", "symbol"]).reset_index(drop=True)


def get_valid_start_timestamps(path: str, n: int = 10000) -> tuple[np.ndarray, list[str], int]:
    """Scan the dataset once to pre-load valid start timestamps and symbol list.

    Returns
    -------
    tuple[np.ndarray, list[str], int]
        (valid_open_times, symbols, k) where k is the per-symbol row count.
    """
    cols = ["symbol", "open_time"]
    if ds is None or pq is None:
        df = pd.read_parquet(path, columns=cols).dropna()
        symbols = _pick_symbols(list(df["symbol"].unique()))
        k = max(1, n // len(symbols))
        df_sub = df[df["symbol"].isin(symbols)]
        anchor_times = np.sort(df_sub[df_sub["symbol"] == symbols[0]]["open_time"].unique())
        return anchor_times, symbols, k

    dataset = ds.dataset(path, format="parquet")
    available_in_file = set(
        dataset.to_table(columns=["symbol"]).column("symbol").unique().to_pylist()
    )
    symbols = [s for s in DEFAULT_SYMBOLS if s in available_in_file]
    if not symbols:
        raise ValueError(f"None of {DEFAULT_SYMBOLS} were found in {path}.")

    k = max(1, n // len(symbols))
    start_times_map = _get_symbol_start_times(dataset, symbols)
    anchor_symbol = max(symbols, key=lambda s: start_times_map.get(s, 0))

    anchor_filter = ds.field("symbol") == anchor_symbol
    valid_open_times = (
        dataset.to_table(columns=["open_time"], filter=anchor_filter)
        .column("open_time")
        .to_numpy()
    )

    if len(valid_open_times) == 0:
        raise ValueError(f"Anchor symbol '{anchor_symbol}' returned 0 timestamps!")

    return valid_open_times, symbols, k


def read_window_from_timestamps(
    path: str,
    valid_open_times: np.ndarray,
    symbols: list[str],
    k: int,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a random window given pre-loaded valid start timestamps and metadata."""
    if cols is None:
        cols = ["symbol", "open_time", "close"]

    t_start, t_end = _random_window(valid_open_times, k)

    if ds is None or pq is None:
        df = pd.read_parquet(path, columns=cols).dropna()
        df_sub = df[df["symbol"].isin(symbols)]
        mask = (df_sub["open_time"] >= t_start) & (df_sub["open_time"] <= t_end)
        return df_sub[mask].sort_values(by=["open_time", "symbol"]).reset_index(drop=True)

    dataset = ds.dataset(path, format="parquet")
    query_filter = (
        ds.field("symbol").isin(symbols)
        & (ds.field("open_time") >= t_start)
        & (ds.field("open_time") <= t_end)
    )
    df = dataset.to_table(columns=cols, filter=query_filter).to_pandas()
    return df.dropna().sort_values(by=["open_time", "symbol"]).reset_index(drop=True)


def _read_last_n_pandas(path: str, n: int, cols: list[str]) -> pd.DataFrame:
    """Fallback path used when PyArrow is unavailable."""
    anchor_times, symbols, k = get_valid_start_timestamps(path, n)
    return read_window_from_timestamps(path, anchor_times, symbols, k, cols)


def _read_last_n_pyarrow(path: str, n: int, cols: list[str]) -> pd.DataFrame:
    """Memory-efficient path with dynamic, metadata-driven time anchoring."""
    valid_open_times, symbols, k = get_valid_start_timestamps(path, n)
    return read_window_from_timestamps(path, valid_open_times, symbols, k, cols)


def read_train_test(path, n_train, n_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with non-overlapping time windows."""
    df = read_last_n(path, n=n_train + n_test)
    # Use last n_test rows as held-out set, always
    split = len(df["open_time"].unique()) - n_test // len(DEFAULT_SYMBOLS)
    times = sorted(df["open_time"].unique())
    cut = times[split]
    return df[df["open_time"] < cut], df[df["open_time"] >= cut]
