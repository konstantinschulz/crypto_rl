"""Log-return reward (ROADMAP P1-1).

The step reward is log(PV_t / PV_{t-1}). Fees are inside PV and counted once,
there is no terminal lump sum, and there are no rule penalties.
"""

import unittest

import numpy as np

from crypto_rl.config import RLConfig
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from tests.test_baselines import _make_arrays

NUM_ASSETS = 2
WINDOW = 120
# max_asset_allocation=1.0 and weight_step=0.05: bin k is a target of 5k %
N_BINS = 21
CASH = [0] * NUM_ASSETS
ALL_IN_0 = [N_BINS - 1, 0]


def _build_env(
    prices=None, n_bars: int = 800, reward_type: str = "log_return", **config_kwargs
) -> MinimalCryptoEnv:
    rand_prices, static, norm_vol = _make_arrays(0, n_bars, NUM_ASSETS)
    if prices is None:
        prices = rand_prices
    # One bar per step, so these tests can check per-bar mechanics;
    # tests/test_decision_interval.py covers longer intervals
    config_kwargs.setdefault("decision_interval", 1)
    # parquet_path=None keeps reset() from reloading the real dataset
    config = RLConfig(
        disable_logging=True,
        parquet_path=None,
        window_size=WINDOW,
        reward_type=reward_type,
        max_asset_allocation=1.0,
        **config_kwargs,
    )
    env = MinimalCryptoEnv(
        prices_arr=prices,
        static_obs=static[: len(prices)],
        norm_vol_arr=norm_vol[: len(prices)],
        asset_names=[f"A{i}" for i in range(NUM_ASSETS)],
        config=config,
        is_eval=True,
    )
    env.reset(seed=0)
    return env


class LogReturnRewardTest(unittest.TestCase):
    def test_rewards_telescope_to_episode_log_return(self):
        # Random trading: the per-step rewards must sum to the episode's
        # mark-to-market log return, with no extra terminal term on top.
        env = _build_env()
        rng = np.random.default_rng(1)
        total, truncated = 0.0, False
        while not truncated:
            action = rng.integers(0, N_BINS, size=NUM_ASSETS)
            _, reward, _, truncated, _ = env.step(action)
            total += reward
        self.assertGreater(env.trades_count, 10)
        self.assertGreater(env.fees_paid_total, 0.0)
        expected = np.log(env.portfolio_value / env.config.budget_initial)
        self.assertAlmostEqual(total, expected, places=6)

    def test_fees_are_counted_once(self):
        # Flat prices: the only PV change on a buy is the fee itself.
        prices = np.full((WINDOW + 10, NUM_ASSETS), 100.0)
        env = _build_env(prices=prices)
        pv0 = env.portfolio_value
        _, reward, _, _, info = env.step(np.array([10, 0]))  # 50% in asset 0
        fee = env.fees_paid_total
        self.assertGreater(fee, 0.0)
        self.assertAlmostEqual(reward, np.log((pv0 - fee) / pv0), places=6)
        self.assertEqual(
            set(info["reward_components"]),
            {"log_return", "benchmark", "drawdown_penalty", "turnover_penalty"},
        )

    def test_all_cash_target_from_cash_is_a_no_op(self):
        env = _build_env()
        _, reward, _, _, _ = env.step(np.array(CASH))
        self.assertEqual(reward, 0.0)
        self.assertEqual(env.trades_count, 0)

    def test_excess_log_return_subtracts_unfloored_basket(self):
        env = _build_env(reward_type="excess_log_return")
        for _ in range(50):
            c = env.current_step
            gross = env.prices_arr[c] / env.prices_arr[c - 1]
            _, reward, _, _, info = env.step(np.array(CASH))
            # All cash: the reward is minus the basket's log return, which is
            # positive whenever the basket falls (no max(0, .) floor).
            self.assertAlmostEqual(reward, -np.log(gross.mean()), places=9)
            self.assertAlmostEqual(
                info["reward_components"]["benchmark"], -np.log(gross.mean())
            )

    def test_drawdown_term_is_opt_in(self):
        prices = np.full((WINDOW + 10, NUM_ASSETS), 100.0)
        prices[WINDOW + 1 :] = 50.0  # bar WINDOW -> WINDOW+1 halves both assets
        env_off = _build_env(prices=prices)
        env_on = _build_env(prices=prices, drawdown_penalty_coef=0.5)
        for env in (env_off, env_on):
            env.step(np.array(ALL_IN_0))  # all-in on asset 0 at 100
        _, _, _, _, info_off = env_off.step(np.array(ALL_IN_0))
        _, _, _, _, info_on = env_on.step(np.array(ALL_IN_0))
        self.assertEqual(info_off["reward_components"]["drawdown_penalty"], 0.0)
        self.assertLess(info_on["reward_components"]["drawdown_penalty"], -0.2)


if __name__ == "__main__":
    unittest.main()
