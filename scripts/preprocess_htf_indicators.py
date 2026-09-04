"""
scripts/preprocess_htf_indicators.py
====================================
Preprocesses 1-minute crypto Parquet data to compute Higher Time Frame (HTF)
indicators (15m slope, 1h slope, 24h daily trend regime) and writes a new
Parquet dataset with row-group-level ZSTD compression for fast PyArrow reading.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def compute_htf_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Higher Time Frame (HTF) indicators on 1m OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'close' column, sorted by time.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - 'htf_slope_15m': 15-minute EMA slope relative to price (15-period lookback)
        - 'htf_slope_1h': 1-hour EMA slope relative to price (60-period lookback)
        - 'htf_regime_24h': 24-hour daily trend regime (relative distance from 1440-minute EMA)
    """
    close = df["close"].astype(np.float32)

    # Calculate Exponential Moving Averages for 15m, 1h (60m), and 24h (1440m)
    ema_15 = close.ewm(span=15, adjust=False).mean()
    ema_60 = close.ewm(span=60, adjust=False).mean()
    ema_1440 = close.ewm(span=1440, adjust=False).mean()

    # 1. 15-minute MA slope (pct change of 15m trend over 15 bars)
    slope_15m = ((ema_15 - ema_15.shift(15)) / (close + 1e-8)).fillna(0.0).astype(np.float32)

    # 2. 1-hour MA slope (pct change of 1h trend over 60 bars)
    slope_1h = ((ema_60 - ema_60.shift(60)) / (close + 1e-8)).fillna(0.0).astype(np.float32)

    # 3. 24-hour daily trend regime (relative distance from 24h EMA)
    regime_24h = ((close - ema_1440) / (ema_1440 + 1e-8)).fillna(0.0).astype(np.float32)

    df["htf_slope_15m"] = slope_15m
    df["htf_slope_1h"] = slope_1h
    df["htf_regime_24h"] = regime_24h

    return df


def preprocess_parquet(
    input_path: str,
    output_path: str,
    row_group_size: int = 43200,
    compression: str = "zstd",
) -> None:
    """Read input parquet file, compute HTF indicators per symbol, and write output parquet."""
    in_p = Path(input_path)
    out_p = Path(output_path)

    if not in_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if in_p.resolve() == out_p.resolve():
        raise ValueError("Output path cannot be identical to input path. Specify a distinct copy.")

    print(f"Opening input dataset: {input_path}")
    dataset = ds.dataset(str(in_p), format="parquet")

    # Get symbol list
    symbols = dataset.to_table(columns=["symbol"]).column("symbol").unique().to_pylist()
    total_symbols = len(symbols)
    print(f"Found {total_symbols} symbols: {symbols[:10]} ...")

    writer = None
    target_schema = None
    total_rows = 0
    t0 = time.time()

    try:
        for idx, symbol in enumerate(symbols, 1):
            sym_t0 = time.time()
            sym_filter = ds.field("symbol") == symbol
            table = dataset.to_table(filter=sym_filter)
            df = table.to_pandas()

            # Ensure chronological order
            if "open_time" in df.columns and not df["open_time"].is_monotonic_increasing:
                df = df.sort_values(by="open_time").reset_index(drop=True)

            # Compute HTF indicators
            df = compute_htf_indicators(df)

            # Ensure proper float32 casting for numeric columns
            float_cols = ["open", "high", "low", "close", "volume", "htf_slope_15m", "htf_slope_1h", "htf_regime_24h"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)

            # Convert to PyArrow table
            out_table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                target_schema = out_table.schema
                print(f"Target Schema:\n{target_schema}")
                writer = pq.ParquetWriter(
                    str(out_p),
                    schema=target_schema,
                    compression=compression,
                    use_dictionary=["symbol"],
                )

            # Write in row groups
            n_rows_sym = len(df)
            for rg_start in range(0, n_rows_sym, row_group_size):
                rg_end = min(rg_start + row_group_size, n_rows_sym)
                chunk = out_table.slice(rg_start, rg_end - rg_start)
                writer.write_table(chunk)

            total_rows += n_rows_sym
            sym_dur = time.time() - sym_t0
            elapsed = time.time() - t0
            print(
                f"[{idx:02d}/{total_symbols:02d}] {symbol:10s}: {n_rows_sym:8,d} rows processed "
                f"({sym_dur:.2f}s, total elapsed: {elapsed:.1f}s)"
            )

    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - t0
    out_size_gb = out_p.stat().st_size / (1024**3)
    print(
        f"\nSuccessfully generated: {output_path}\n"
        f"Total rows: {total_rows:,}\n"
        f"Output size: {out_size_gb:.2f} GB\n"
        f"Total duration: {total_time:.1f} seconds ({total_rows/total_time:,.0f} rows/s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess 1m Parquet dataset with HTF indicators.")
    parser.add_argument(
        "--input",
        type=str,
        default="binance_spot_1m_last4y_single.parquet",
        help="Path to source Parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="binance_spot_1m_last4y_single_htf.parquet",
        help="Path to target augmented Parquet dataset.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=43200,
        help="Row group size for chunked writing (default: 43200 = ~1 month of 1m bars).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
        help="Parquet compression codec (default: zstd).",
    )

    args = parser.parse_args()
    preprocess_parquet(
        input_path=args.input,
        output_path=args.output,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
