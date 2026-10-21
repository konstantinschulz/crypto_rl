"""The BTC regime lags must cover 24 hours and 7 days of 1-minute bars."""

import unittest

import numpy as np

from crypto_rl.env.feature_utils import (
    REGIME_LAG_7D,
    REGIME_LAG_24H,
    btc_regime_one_hot,
)

BULL, RANGING, BEAR = 0, 1, 2


class RegimeFeatureTest(unittest.TestCase):
    def test_lags_are_in_minutes(self):
        self.assertEqual(REGIME_LAG_24H, 24 * 60)
        self.assertEqual(REGIME_LAG_7D, 7 * 24 * 60)

    def test_warmup_is_ranging(self):
        prices = np.linspace(100.0, 200.0, REGIME_LAG_7D + 10)
        regime = btc_regime_one_hot(prices)
        np.testing.assert_array_equal(regime.sum(axis=1), 1.0)
        self.assertTrue((regime[:REGIME_LAG_7D, RANGING] == 1.0).all())
        self.assertTrue((regime[REGIME_LAG_7D:, BULL] == 1.0).all())

    def test_short_moves_do_not_flip_the_regime(self):
        # A steady 7-day uptrend with a 3-hour dip of 2%: the old 24- and
        # 168-bar lags called the dip a bear market, 24h/7d lags must not.
        n = REGIME_LAG_7D + 2 * REGIME_LAG_24H
        prices = 100.0 * np.exp(np.linspace(0.0, 0.3, n))
        dip = slice(n - 300, n - 120)
        prices[dip] *= 0.98
        regime = btc_regime_one_hot(prices)
        self.assertEqual(regime[dip, BEAR].sum(), 0.0)
        self.assertTrue((regime[dip, BULL] == 1.0).all())

    def test_downtrend_is_bear(self):
        n = REGIME_LAG_7D + REGIME_LAG_24H
        prices = 100.0 * np.exp(np.linspace(0.0, -0.3, n))
        regime = btc_regime_one_hot(prices)
        self.assertTrue((regime[REGIME_LAG_7D:, BEAR] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
