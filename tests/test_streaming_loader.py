"""Regression tests for the streaming window loader.

Symbols are interleaved in the Parquet file and the symbol column is
dictionary-encoded, as in the real dataset. These are the two conditions under
which the per-symbol loader previously blew up memory or returned wrong symbols.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from crypto_rl.data import load_window_wide
from crypto_rl.env.data_utils import pivot_ohlcv, pivot_ohlcv_wide

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None

SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
COLS = ["close", "open", "high", "low", "volume"]


def _write_interleaved_parquet(path: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for t in range(400):
        for s in SYMBOLS:
            close = 100 + rng.normal()
            rows.append(
                {
                    "symbol": s,
                    "open_time": 1_700_000_000_000 + t * 60_000,
                    "close": close,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "volume": abs(rng.normal()) + 1.0,
                }
            )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    # One missing value, to exercise the dropna / ffill path
    df.loc[df.index[5], "close"] = np.nan
    table = pa.Table.from_pandas(
        df.assign(symbol=df["symbol"].astype("category")), preserve_index=False
    )
    pq.write_table(table, path, row_group_size=97)
    return df


@unittest.skipIf(pa is None, "pyarrow not installed")
class StreamingLoaderTest(unittest.TestCase):
    def test_wide_loader_matches_long_pivot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "interleaved.parquet")
            raw = _write_interleaved_parquet(path)

            t_start = raw["open_time"].min() + 60_000 * 10
            t_end = raw["open_time"].min() + 60_000 * 300
            window = raw[(raw["open_time"] >= t_start) & (raw["open_time"] <= t_end)]

            wide = load_window_wide(path, t_start, t_end, SYMBOLS, cols=None)
            self.assertEqual(set(wide), set(COLS))
            for col in COLS:
                self.assertEqual(
                    sorted(wide[col].columns), sorted(SYMBOLS), msg=f"columns for {col}"
                )

            long_df = window.dropna().copy()
            long_df = long_df.astype({c: np.float32 for c in COLS})
            expected = pivot_ohlcv(long_df.copy())
            got = pivot_ohlcv_wide(wide)

            names = ["close", "open", "high", "low", "volume"]
            for name, exp, act in zip(names, expected[:5], got[:5]):
                exp = exp.sort_index(axis=1)
                act = act.reindex(index=exp.index, columns=exp.columns)
                np.testing.assert_allclose(
                    act.to_numpy(dtype=float),
                    exp.to_numpy(dtype=float),
                    rtol=1e-6,
                    equal_nan=True,
                    err_msg=name,
                )


if __name__ == "__main__":
    unittest.main()
