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
import pyarrow.dataset as ds

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
    cols = ["symbol", "open_time", "close", "open", "high", "low", "volume"]

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
    df = _downcast_ohlcv(df)
    symbols = _pick_symbols(list(df["symbol"].unique()))
    k = max(1, n // len(symbols))

    df_sub = df[df["symbol"].isin(symbols)]
    anchor_times = np.sort(df_sub[df_sub["symbol"] == symbols[0]]["open_time"].unique())
    t_start, t_end = _random_window(anchor_times, k)

    mask = (df_sub["open_time"] >= t_start) & (df_sub["open_time"] <= t_end)
    return df_sub[mask].sort_values(by=["open_time", "symbol"]).reset_index(drop=True)


def get_valid_start_timestamps(
    path: str, n: int = 10000
) -> tuple[np.ndarray, list[str], int]:
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
        anchor_times = np.sort(
            df_sub[df_sub["symbol"] == symbols[0]]["open_time"].unique()
        )
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
    valid_open_times = np.sort(np.unique(valid_open_times))  # Sort & deduplicate
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
        cols = ["symbol", "open_time", "close", "open", "high", "low", "volume"]

    t_start, t_end = _random_window(valid_open_times, k)

    if ds is None or pq is None:
        df = pd.read_parquet(path, columns=cols).dropna()
        df = _downcast_ohlcv(df)
        df_sub = df[df["symbol"].isin(symbols)]
        mask = (df_sub["open_time"] >= t_start) & (df_sub["open_time"] <= t_end)
        return (
            df_sub[mask].sort_values(by=["open_time", "symbol"]).reset_index(drop=True)
        )

    dataset = ds.dataset(path, format="parquet")
    query_filter = (
        ds.field("symbol").isin(symbols)
        & (ds.field("open_time") >= t_start)
        & (ds.field("open_time") <= t_end)
    )
    df = dataset.to_table(columns=cols, filter=query_filter).to_pandas()
    df = _downcast_ohlcv(df)
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
    """Return (train_df, test_df) with a single train/test split using memory-efficient PyArrow reads."""
    total = n_train + n_test
    df = read_n_rows(path, total)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df


def read_n_rows(path: str, n_rows: int) -> pd.DataFrame:
    """Load the most recent n_rows sorted chronologically with memory-efficient PyArrow reads."""
    cols = ["symbol", "open_time", "close", "open", "high", "low", "volume"]

    try:
        dataset = ds.dataset(path, format="parquet")

        # 1. Identify available symbols and pick the target trading universe
        available_symbols = (
            dataset.to_table(columns=["symbol"]).column("symbol").unique().to_pylist()
        )
        symbols = _pick_symbols(available_symbols)
        symbol_filter = ds.field("symbol").isin(symbols)

        # 2. Get timestamps for an anchor symbol within the target group to calculate the exact cutoff
        anchor_symbol = symbols[0]
        anchor_filter = symbol_filter & (ds.field("symbol") == anchor_symbol)
        anchor_times = (
            dataset.to_table(columns=["open_time"], filter=anchor_filter)
            .column("open_time")
            .to_numpy()
        )
        anchor_times = np.sort(anchor_times)

        # Calculate required timesteps (k) per symbol to achieve `n_rows` globally
        k = max(1, n_rows // len(symbols))

        if len(anchor_times) > k:
            cutoff_time = anchor_times[-k]
        else:
            cutoff_time = anchor_times[0]

        # 3. Push BOTH symbol and time filters down to PyArrow
        query_filter = symbol_filter & (ds.field("open_time") >= cutoff_time)
        df = dataset.to_table(columns=cols, filter=query_filter).to_pandas()

        # Ensure strict sorting and trim to exact row count
        df = df.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)
        df = df.iloc[-n_rows:]

    except Exception as e:
        print(f"PyArrow optimized read failed ({e}), falling back to pandas read...")
        df = pd.read_parquet(path, columns=cols).dropna()
        symbols = _pick_symbols(list(df["symbol"].unique()))
        df = df[df["symbol"].isin(symbols)]
        df = df.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)
        df = df.iloc[-n_rows:]

    df = df.dropna()
    df = _downcast_ohlcv(df)
    return df


def get_walk_forward_splits(
    df: pd.DataFrame, n_folds: int = 3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split dataframe into n_folds expanding walk-forward train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe containing at least 'open_time' and 'symbol'.
    n_folds : int
        Number of walk-forward folds (default: 3).

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_df, test_df) tuples for each fold.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    unique_times = np.sort(df["open_time"].unique())
    n_unique = len(unique_times)
    n_chunks = n_folds + 1
    chunk_len = n_unique // n_chunks

    if chunk_len == 0:
        raise ValueError(
            f"Not enough unique timestamps ({n_unique}) to create {n_folds} folds."
        )

    splits = []
    for fold in range(n_folds):
        train_end_idx = (fold + 1) * chunk_len
        test_start_idx = train_end_idx
        test_end_idx = (fold + 2) * chunk_len if fold < (n_folds - 1) else n_unique

        t_train_max = unique_times[train_end_idx - 1]
        t_test_min = unique_times[test_start_idx]
        t_test_max = unique_times[test_end_idx - 1]

        train_df = df[df["open_time"] <= t_train_max].copy().reset_index(drop=True)
        test_df = (
            df[(df["open_time"] >= t_test_min) & (df["open_time"] <= t_test_max)]
            .copy()
            .reset_index(drop=True)
        )
        splits.append((train_df, test_df))

    return splits


_OHLCV_NUMERIC_COLS = ["close", "open", "high", "low", "volume"]


def _downcast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Cast OHLCV numeric columns to float32 in-place to halve memory.

    Only columns that exist in the DataFrame are touched, so this is safe
    to call on any subset of the full schema.
    """
    for col in _OHLCV_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df
