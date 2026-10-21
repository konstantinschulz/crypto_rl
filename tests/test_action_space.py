"""Target-weight action space (ROADMAP P1-3).

The action picks one weight bin per asset. The env rebalances toward the
target, sells first, and skips assets within the no-trade band. Every action
is valid.
"""

import unittest

import numpy as np

from crypto_rl.config import RLConfig
from crypto_rl.env.action_processing import weight_bins
from tests.test_reward import N_BINS, NUM_ASSETS, WINDOW, _build_env

FLAT = np.full((WINDOW + 20, NUM_ASSETS), 100.0)


def _weights(env) -> np.ndarray:
    """Current weights, marked at the price the next step trades at."""
    values = env.holdings * env.prices_arr[env.current_step - 1]
    return values / (env.cash + values.sum())


class WeightBinsTest(unittest.TestCase):
    def test_default_bins(self):
        config = RLConfig()
        np.testing.assert_allclose(
            weight_bins(config), [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        )

    def test_action_space_has_one_bin_choice_per_asset(self):
        env = _build_env()
        np.testing.assert_array_equal(env.action_space.nvec, [N_BINS] * NUM_ASSETS)
        self.assertAlmostEqual(env.weight_bins[-1], 1.0)

    def test_cap_must_be_a_multiple_of_the_step(self):
        with self.assertRaises(ValueError):
            weight_bins(RLConfig(max_asset_allocation=0.22, weight_step=0.05))
        # Float noise from e.g. Optuna's 0.1 + 0.05 * k grid is fine
        self.assertEqual(
            len(weight_bins(RLConfig(max_asset_allocation=0.1 + 0.05 * 3))), 6
        )


class RebalanceTest(unittest.TestCase):
    def test_reaches_target_weights_from_cash(self):
        env = _build_env(prices=FLAT)
        env.step(np.array([10, 6]))  # 50% and 30%
        # Fees come out of PV, so weights land a hair above target
        np.testing.assert_allclose(_weights(env), [0.5, 0.3], atol=1e-3)
        self.assertEqual(env.trades_count, 2)
        self.assertEqual(env.fill_counts["BUY"], 2)
        self.assertAlmostEqual(env.per_asset_fees.sum(), env.fees_paid_total, places=5)

    def test_targets_above_100_percent_are_scaled_down(self):
        env = _build_env(prices=FLAT)
        env.step(np.array([N_BINS - 1, N_BINS - 1]))  # 100% + 100%
        np.testing.assert_allclose(_weights(env), [0.5, 0.5], atol=1e-6)
        self.assertGreaterEqual(env.cash, 0.0)
        self.assertLess(env.cash, 1e-6)

    def test_rotation_sells_first_to_fund_the_buy(self):
        env = _build_env(prices=FLAT)
        env.step(np.array([N_BINS - 1, 0]))  # all-in on asset 0
        env.step(np.array([0, N_BINS - 1]))  # rotate into asset 1
        self.assertEqual(env.holdings[0], 0.0)
        np.testing.assert_allclose(_weights(env), [0.0, 1.0], atol=1e-6)
        self.assertEqual(env.fill_counts["SELL"], 1)
        self.assertEqual(env.fill_counts["BUY"], 2)
        self.assertEqual(env.total_closed_trades, 1)
        self.assertGreaterEqual(env.cash, 0.0)

    def test_no_trade_band(self):
        prices = FLAT.copy()
        # The first step fills at prices[WINDOW - 1]; asset 0 is 3% higher
        # at the second step, so its weight drifts from 50% to about 50.7%,
        # inside the 2% band.
        prices[WINDOW:, 0] = 103.0
        env = _build_env(prices=prices)
        env.step(np.array([10, 10]))
        trades = env.trades_count
        env.step(np.array([10, 10]))
        self.assertEqual(env.trades_count, trades)
        # One bin (5%) is outside the band and trades
        env.step(np.array([9, 10]))
        self.assertEqual(env.trades_count, trades + 1)
        self.assertEqual(env.fill_counts["SELL"], 1)

    def test_zero_target_closes_a_position_inside_the_band(self):
        env = _build_env(prices=FLAT)
        # A 1% position: below the 2% band, but a target of 0 still closes it
        env.holdings[0] = 1.0
        env.cash -= 100.0
        env.avg_entry_price[0] = 100.0
        env.step(np.array([0, 0]))
        self.assertEqual(env.holdings[0], 0.0)
        self.assertEqual(env.fill_counts["SELL"], 1)

    def test_band_zero_trades_every_drift(self):
        prices = FLAT.copy()
        prices[WINDOW:, 0] = 103.0
        env = _build_env(prices=prices, rebalance_band=0.0)
        env.step(np.array([10, 10]))
        trades = env.trades_count
        env.step(np.array([10, 10]))
        self.assertEqual(env.trades_count, trades + 2)

    def test_fill_counts_reset_each_episode(self):
        env = _build_env(prices=FLAT)
        env.step(np.array([10, 10]))
        self.assertEqual(sum(env.fill_counts.values()), 2)
        env.reset(seed=0)
        self.assertEqual(sum(env.fill_counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
