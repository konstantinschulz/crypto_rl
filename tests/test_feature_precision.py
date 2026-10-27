"""Rolling feature helpers stay accurate on long, large-magnitude series.

They difference cumulative sums over the whole slice. With float32
accumulation, BTC-sized prices over ~1M bars turned a window's mean / std /
correlation into noise (macd, btc_corr_60m and vwap_dev before
FEATURE_VERSION 3).
"""

import unittest

import numpy as np
import pandas as pd

from crypto_rl.env.feature_utils import (
    _rolling_corr,
    _rolling_mean_std,
    _rolling_sum,
    _rolling_zscore,
)

N_BARS = 1_000_000


def _random_walk(seed: int, start: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 1e-3, N_BARS)
    return (start * np.exp(np.cumsum(log_returns))).astype(np.float32)


class RollingPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.btc = _random_walk(0, 60_000.0)
        cls.eth = _random_walk(1, 3_000.0)
        cls.tail = slice(N_BARS - 5_000, N_BARS)  # where the sums are largest

    def test_mean_std_match_pandas(self):
        mean, std = _rolling_mean_std(self.btc[:, None], 240)
        series = pd.Series(self.btc.astype(np.float64))
        ref_mean = series.rolling(240).mean().to_numpy()
        ref_std = series.rolling(240).std(ddof=0).to_numpy()
        np.testing.assert_allclose(mean[self.tail, 0], ref_mean[self.tail], rtol=1e-6)
        np.testing.assert_allclose(std[self.tail, 0], ref_std[self.tail], rtol=1e-2)

    def test_sum_matches_pandas(self):
        volume_value = self.btc * 10.0  # price × volume scale
        out = _rolling_sum(volume_value[:, None], 1440)
        ref = pd.Series(volume_value.astype(np.float64)).rolling(1440).sum().to_numpy()
        np.testing.assert_allclose(out[self.tail, 0], ref[self.tail], rtol=1e-5)

    def test_corr_matches_pandas(self):
        out = _rolling_corr(self.eth[:, None], self.btc[:, None], 60)[:, 0]
        ref = (
            pd.Series(self.eth.astype(np.float64))
            .rolling(60)
            .corr(pd.Series(self.btc.astype(np.float64)))
            .to_numpy()
        )
        np.testing.assert_allclose(out[self.tail], ref[self.tail], atol=1e-3)

    def test_zscore_matches_pandas(self):
        momentum = (self.btc / np.roll(self.btc, 240) - 1.0)[240:, None]
        out = _rolling_zscore(momentum, window=100, min_periods=10)[:, 0]
        series = pd.Series(momentum[:, 0].astype(np.float64))
        roll = series.rolling(100)
        ref = ((series - roll.mean()) / roll.std(ddof=0)).to_numpy()
        tail = slice(len(ref) - 5_000, len(ref))
        np.testing.assert_allclose(out[tail], ref[tail], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
