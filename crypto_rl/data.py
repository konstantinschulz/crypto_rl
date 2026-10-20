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

from typing import NamedTuple

import numpy as np
import pandas as pd

# Try to use pyarrow for efficient, row-group-aware parquet reads so we
# don't load the entire file into memory.  Fallback to pandas.read_parquet
# if pyarrow is not available.
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception:
    pa = None  # type: ignore
    pc = None  # type: ignore
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
HTF_COLS = ["htf_slope_15m", "htf_slope_1h", "htf_regime_24h"]

# The source parquet mixes units in `open_time`: older rows are milliseconds,
# recent rows (from 2025-01-01) are microseconds. Anything below this threshold
# is far too small to be microseconds (it would be before 1973), so it is ms.
_US_THRESHOLD = 10**14


def open_time_to_us(ts: int) -> int:
    """Normalise a raw `open_time` value to microseconds since the epoch."""
    ts = int(ts)
    return ts * 1000 if ts < _US_THRESHOLD else ts


def open_time_to_datetime(ts: int) -> pd.Timestamp:
    """Convert a raw `open_time` value to a UTC timestamp, whatever its unit."""
    return pd.Timestamp(open_time_to_us(ts), unit="us", tz="UTC")
_OHLCV_NUMERIC_COLS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "htf_slope_15m",
    "htf_slope_1h",
    "htf_regime_24h",
]


def _get_read_cols(path: str) -> list[str]:
    """Return OHLCV columns augmented with HTF indicator columns if present."""
    base_cols = ["symbol", "open_time", "close", "open", "high", "low", "volume"]
    try:
        if pq is not None:
            schema_names = pq.ParquetFile(path).schema.names
            available_htf = [c for c in HTF_COLS if c in schema_names]
            return base_cols + available_htf
    except Exception:
        pass
    return base_cols


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
    cols = _get_read_cols(path)
    valid_open_times, symbols, k = get_valid_start_timestamps(path, n)
    return read_window_from_timestamps(path, valid_open_times, symbols, k, cols)


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
        cols = _get_read_cols(path)

    t_start, t_end = _random_window(valid_open_times, k)

    if ds is None or pq is None:
        df = pd.read_parquet(path, columns=cols)
        mask = (
            (df["symbol"].isin(symbols))
            & (df["open_time"] >= t_start)
            & (df["open_time"] <= t_end)
        )
        df = df[mask]
    else:
        dataset = ds.dataset(path, format="parquet")
        query_filter = (
            ds.field("symbol").isin(symbols)
            & (ds.field("open_time") >= t_start)
            & (ds.field("open_time") <= t_end)
        )
        df = dataset.to_table(columns=cols, filter=query_filter).to_pandas()

    # Unified post-processing
    df = df.dropna()
    df = _downcast_ohlcv(df)
    return df.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)


def load_window_wide(
    path: str,
    t_start,
    t_end,
    symbols: list[str],
    cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a time window as wide float32 frames, one symbol at a time.

    Unlike ``read_window_from_timestamps`` this never materialises the
    long-format window (symbols x rows) in memory. Each symbol is read,
    cleaned and pivotted into its own column before the next is read, so the
    peak cost is one symbol's rows plus the wide result.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each value column (``close``, ``open``, ...) to a frame indexed
        by ``open_time`` with one column per symbol, sorted by symbol. NaNs
        are left in place for the caller to forward/back-fill.
    """
    if cols is None:
        cols = _get_read_cols(path)
    # The symbol is used for grouping only; drop it from the per-symbol frames
    # to avoid carrying millions of identical Python strings.
    read_cols = list(dict.fromkeys(["symbol", *cols]))
    value_cols = [c for c in cols if c not in ("symbol", "open_time")]
    wanted = set(symbols)

    # One streaming pass over the file. A symbol filter cannot skip row groups
    # because symbols are interleaved in the file, and a per-symbol Arrow scan
    # buffers ~2.8 GB for a 2M-row window. Batches of 1M rows keep the scan's
    # working set small; only the selected rows are kept, downcast to float32.
    pieces: dict[str, list[pd.DataFrame]] = {s: [] for s in wanted}
    if pq is not None:
        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(batch_size=1 << 20, columns=read_cols)
        for batch in batches:
            table = pa.Table.from_batches([batch])
            ot = table.column("open_time")
            in_window = pc.and_(
                pc.greater_equal(ot, t_start), pc.less_equal(ot, t_end)
            )
            kept = table.filter(in_window).to_pandas()
            del table, batch
            kept = kept[kept["symbol"].isin(wanted)]
            if kept.empty:
                continue
            for symbol, group in kept.groupby("symbol", sort=False, observed=True):
                pieces[symbol].append(
                    _downcast_ohlcv(group.drop(columns="symbol"))
                )
            del kept
    else:  # pragma: no cover - pyarrow missing; full read is the only option
        full = pd.read_parquet(path, columns=read_cols)
        full = full[full["symbol"].isin(wanted) & full["open_time"].between(t_start, t_end)]
        for symbol, group in full.groupby("symbol", sort=False, observed=True):
            pieces[symbol].append(_downcast_ohlcv(group.drop(columns="symbol")))
        del full

    per_col: dict[str, dict[str, pd.Series]] = {c: {} for c in value_cols}
    for symbol in sorted(symbols):
        if not pieces[symbol]:
            continue
        sym_df = pd.concat(pieces[symbol], ignore_index=True)
        pieces[symbol] = []
        # Same NaN handling as the long-format path, applied per symbol:
        # a row with any NaN is dropped and later filled by ffill/bfill.
        sym_df = sym_df.dropna()
        sym_df = (
            sym_df.drop_duplicates("open_time")
            .sort_values("open_time")
            .set_index("open_time")
        )
        for col in value_cols:
            if col in sym_df.columns:
                per_col[col][symbol] = sym_df[col]
        del sym_df

    return {
        col: pd.DataFrame(series_by_symbol).sort_index()
        for col, series_by_symbol in per_col.items()
        if series_by_symbol
    }


def load_random_window_wide(
    path: str,
    valid_open_times: np.ndarray,
    symbols: list[str],
    k: int,
    cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Random-window counterpart of :func:`load_window_wide`."""
    t_start, t_end = _random_window(valid_open_times, k)
    return load_window_wide(path, t_start, t_end, symbols, cols)


def read_train_test(path, n_train, n_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with a single train/test split using memory-efficient PyArrow reads."""
    total = n_train + n_test
    df = read_n_rows(path, total)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df


def _read_filtered_batches(
    path: str,
    cols: list[str],
    symbols: list[str],
    t_min,
    t_max,
) -> pd.DataFrame:
    """Read rows for ``symbols`` with ``open_time`` in ``[t_min, t_max]``.

    Streams the file in 1M-row batches, keeps only matching rows, and
    downcasts their OHLCV columns to float32 per batch. Peak memory is the
    kept rows plus one batch, not the whole Arrow result.
    """
    read_cols = list(dict.fromkeys(["symbol", *cols]))
    wanted = set(symbols)
    pieces: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1 << 20, columns=read_cols
    ):
        table = pa.Table.from_batches([batch])
        in_window = pc.greater_equal(table.column("open_time"), t_min)
        if t_max is not None:
            in_window = pc.and_(in_window, pc.less_equal(table.column("open_time"), t_max))
        kept = table.filter(in_window).to_pandas()
        del table, batch
        kept = kept[kept["symbol"].isin(wanted)]
        if not kept.empty:
            pieces.append(_downcast_ohlcv(kept))
    if not pieces:
        return pd.DataFrame(columns=cols)
    df = pd.concat(pieces, ignore_index=True)
    if not isinstance(df["symbol"].dtype, pd.CategoricalDtype):
        df["symbol"] = df["symbol"].astype("category")
    return df


def read_n_rows(path: str, n_rows: int) -> pd.DataFrame:
    """Load the most recent n_rows sorted chronologically with memory-efficient PyArrow reads."""
    cols = _get_read_cols(path)

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

        # 3. Stream batches and keep only the selected rows, so the full
        # Arrow result (symbols x rows) is never held in memory at once.
        df = _read_filtered_batches(path, cols, symbols, cutoff_time, None)

    except Exception as e:
        print(f"PyArrow optimized read failed ({e}), falling back to pandas read...")
        df = pd.read_parquet(path, columns=cols)
        symbols = _pick_symbols(list(df["symbol"].unique()))
        df = df[df["symbol"].isin(symbols)]

    # Unified post-processing
    df = df.dropna()
    df = _downcast_ohlcv(df)
    df = df.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)
    return df.iloc[-n_rows:]


class FoldSplit(NamedTuple):
    """One fold's time bounds (inclusive ``open_time`` values).

    train: [data start, t_train_max]
    val:   [t_val_min, t_val_max]   directly after train
    test:  [t_test_min, t_test_max] directly after val
    """

    t_train_max: int
    t_val_min: int
    t_val_max: int
    t_test_min: int
    t_test_max: int
    train_start_str: str
    train_end_str: str


def make_walk_forward_splits(
    unique_times: np.ndarray,
    n_folds: int,
    test_fraction: float,
    val_fraction: float,
) -> list[FoldSplit]:
    """Split sorted unique timestamps into train / val / test per fold.

    With ``n_folds == 1`` the last ``test_fraction`` of the data is the test
    slice. With more folds the data is cut into ``n_folds + 1`` equal chunks and
    fold ``i`` tests on chunk ``i + 1`` (expanding window). In both cases the
    validation slice is the ``val_fraction`` of the data directly before the
    test slice, and the training slice is everything before the validation
    slice, so the test slices do not depend on ``val_fraction``.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    unique_times = np.sort(np.asarray(unique_times))
    n_unique = len(unique_times)
    n_val = round(n_unique * val_fraction)
    if n_val < 1:
        raise ValueError(
            f"val_fraction={val_fraction} yields an empty validation slice "
            f"for {n_unique} timestamps."
        )

    # (test_start_idx, test_end_idx) inclusive, per fold
    if n_folds == 1:
        n_test = round(n_unique * test_fraction)
        if n_test < 1:
            raise ValueError(
                f"test_fraction={test_fraction} yields an empty test slice "
                f"for {n_unique} timestamps."
            )
        test_bounds = [(n_unique - n_test, n_unique - 1)]
    else:
        chunk_len = n_unique // (n_folds + 1)
        if chunk_len == 0:
            raise ValueError(
                f"Not enough unique timestamps ({n_unique}) to create {n_folds} folds."
            )
        test_bounds = [
            (
                (fold + 1) * chunk_len,
                (fold + 2) * chunk_len - 1 if fold < n_folds - 1 else n_unique - 1,
            )
            for fold in range(n_folds)
        ]

    splits = []
    for fold, (test_start, test_end) in enumerate(test_bounds):
        val_start = test_start - n_val
        train_end = val_start - 1
        if train_end < 0:
            raise ValueError(
                f"Fold {fold}: val_fraction={val_fraction} leaves no training "
                f"data before the validation slice."
            )
        splits.append(
            FoldSplit(
                t_train_max=int(unique_times[train_end]),
                t_val_min=int(unique_times[val_start]),
                t_val_max=int(unique_times[test_start - 1]),
                t_test_min=int(unique_times[test_start]),
                t_test_max=int(unique_times[test_end]),
                # Format strings here so we never need the DataFrame later
                train_start_str=open_time_to_datetime(unique_times[0]).strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
                train_end_str=open_time_to_datetime(unique_times[train_end]).strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
            )
        )
    return splits


def get_walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> list[FoldSplit]:
    """Split a dataframe into expanding walk-forward train / val / test folds.

    See :func:`make_walk_forward_splits` for the layout.
    """
    return make_walk_forward_splits(
        df["open_time"].unique(), n_folds, test_fraction, val_fraction
    )


def _downcast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Cast OHLCV numeric columns to float32 in-place to halve memory.

    Only columns that exist in the DataFrame are touched, so this is safe
    to call on any subset of the full schema.
    """
    for col in _OHLCV_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df
