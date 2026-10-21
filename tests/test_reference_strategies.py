"""Rule-based reference strategies (ROADMAP P2-2).

They must only read bars up to the last closed one, map their selection onto
the target-weight action space, and run through the same env, fees and window
as the agent.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from crypto_rl.config import RLConfig
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM
from crypto_rl.experiment.fold_runner import _eval_env_args, run_reference_strategies
from crypto_rl.experiment.reference_strategies import (
    BasketTrendFilter,
    TimeSeriesMomentum,
)

TOP_BIN = 4  # weight_bins 0, 0.05, ..., 0.20


def _fake_env(prices: np.ndarray, current_step: int):
    return SimpleNamespace(
        prices_arr=prices,
        current_step=current_step,
        num_assets=prices.shape[1],
        weight_bins=np.arange(TOP_BIN + 1) * 0.05,
        cash=100.0,
        portfolio_value=100.0,
    )


def _action(strategy, prices: np.ndarray, current_step: int) -> np.ndarray:
    strategy.attach(_fake_env(prices, current_step))
    action, _ = strategy.predict(obs=None)
    return action[0]


class SelectionTest(unittest.TestCase):
    def test_momentum_holds_assets_that_rose_over_the_lookback(self):
        prices = np.array(
            [
                [100.0, 100.0, 100.0],
                [110.0, 90.0, 0.0],  # asset 2 has no valid price
                [999.0, 999.0, 999.0],  # bar t: not closed at the decision
            ]
        )
        action = _action(TimeSeriesMomentum(lookback=1), prices, current_step=2)
        np.testing.assert_array_equal(action, [TOP_BIN, 0, 0])

    def test_momentum_uses_fixed_sleeves_of_at_most_one_over_n(self):
        prices = np.ones((3, 9))
        prices[2, :4] = 2.0  # four assets rose
        action = _action(TimeSeriesMomentum(lookback=1), prices, current_step=3)
        # Largest bin <= 1/9 is 0.10 (bin 2), whatever the number selected
        np.testing.assert_array_equal(action, [2] * 4 + [0] * 5)

    def test_basket_trend_is_all_or_nothing(self):
        up = np.array([[100.0, 10.0], [101.0, 10.0], [103.0, 11.0]])
        np.testing.assert_array_equal(
            _action(BasketTrendFilter(lookback=2), up, current_step=3),
            [TOP_BIN, TOP_BIN],
        )
        down = up[::-1].copy()
        np.testing.assert_array_equal(
            _action(BasketTrendFilter(lookback=2), down, current_step=3), [0, 0]
        )

    def test_holds_cash_until_the_lookback_is_filled(self):
        prices = np.linspace(100.0, 200.0, 50)[:, None]
        for strategy in (TimeSeriesMomentum(lookback=10), BasketTrendFilter(10)):
            self.assertEqual(_action(strategy, prices, current_step=10)[0], 0)
            self.assertEqual(_action(strategy, prices, current_step=11)[0], TOP_BIN)

    def test_never_reads_the_current_or_later_bars(self):
        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 3)), axis=0))
        for strategy in (TimeSeriesMomentum(60), BasketTrendFilter(60)):
            for t in range(61, 400, 7):
                future_hidden = prices.copy()
                future_hidden[t:] = np.nan
                np.testing.assert_array_equal(
                    _action(strategy, prices, t), _action(strategy, future_hidden, t)
                )


class EpisodeTest(unittest.TestCase):
    def test_always_invested_single_asset_matches_buy_and_hold(self):
        # A steadily rising asset is always selected: the first decision buys
        # with all cash and the book never trades again, so the result must
        # equal the buy-and-hold baseline over the same window and fees.
        config = RLConfig(
            disable_logging=True,
            parquet_path=None,
            n_rows=0,
            window_size=120,
            max_asset_allocation=1.0,
        )
        n_bars = 3000
        prices = np.linspace(100.0, 130.0, n_bars + 1)[:, None]
        static = np.zeros((n_bars + 1, MACRO_DIM + STATIC_PER_ASSET_DIM))
        norm_vol = np.full((n_bars + 1, 1), 1e-3)
        env_args = _eval_env_args(
            (prices, static, norm_vol, ["BTCUSDT"]),
            run_id="t",
            is_fast_eval=False,
            config=config,
            logger=None,
        )
        results = run_reference_strategies(
            env_args, {"momentum": TimeSeriesMomentum(lookback=60)}
        )
        r = results["momentum"]
        self.assertAlmostEqual(
            r["final_portfolio_value"] / r["baselines"]["btc"], 1.0, places=5
        )
        self.assertEqual(r["trades_count"], 1)
        self.assertGreater(r["mean_exposure"], 0.99)


if __name__ == "__main__":
    unittest.main()
