"""Decision interval (ROADMAP P2-1).

The agent decides every ``decision_interval`` one-minute bars. Each step
rebalances once, then holds the book bar by bar, so a k-bar step must match k
one-bar steps that hold the same book.
"""

import unittest

import numpy as np

from crypto_rl.env.metrics import get_calmar_from_portfolio_series
from tests.test_reward import ALL_IN_0, CASH, N_BINS, NUM_ASSETS, WINDOW, _build_env

K = 15


def _run(env, action) -> list[tuple[float, dict]]:
    """Step with a fixed action until the episode ends."""
    out, truncated = [], False
    while not truncated:
        _, reward, _, truncated, info = env.step(np.array(action))
        out.append((reward, info["reward_components"]))
    return out


class DecisionIntervalTest(unittest.TestCase):
    def test_steps_advance_k_bars_and_the_last_one_is_shorter(self):
        env = _build_env(decision_interval=K)
        start, end = env.current_step, env.max_steps
        steps = _run(env, CASH)
        self.assertEqual(len(steps), -(-(end - start) // K))
        self.assertEqual(env.current_step, end)
        # One PV mark per bar, not per decision
        self.assertEqual(len(env.portfolio_values), end - start)

    def test_matches_one_bar_steps_holding_the_same_book(self):
        # rebalance_band=1.0: the 1-bar env buys once and then never trades,
        # so both envs hold the same book over the same bars.
        kwargs = {"reward_type": "excess_log_return", "drawdown_penalty_coef": 0.5}
        env_k = _build_env(decision_interval=K, **kwargs)
        env_1 = _build_env(decision_interval=1, rebalance_band=1.0, **kwargs)
        steps_k = _run(env_k, ALL_IN_0)
        steps_1 = _run(env_1, ALL_IN_0)
        self.assertEqual(env_k.trades_count, 1)
        self.assertEqual(env_1.trades_count, 1)

        np.testing.assert_allclose(
            env_k.portfolio_values, env_1.portfolio_values, rtol=1e-9
        )
        for i, (reward, comps) in enumerate(steps_k):
            chunk = steps_1[i * K : (i + 1) * K]
            self.assertAlmostEqual(reward, sum(r for r, _ in chunk), places=9)
            for name, value in comps.items():
                self.assertAlmostEqual(
                    value, sum(c[name] for _, c in chunk), places=9, msg=name
                )
        # The drawdown term is tracked per bar inside the interval
        self.assertLess(sum(c["drawdown_penalty"] for _, c in steps_k), 0.0)

    def test_rewards_telescope_with_random_trading(self):
        env = _build_env(decision_interval=K)
        rng = np.random.default_rng(1)
        total, truncated = 0.0, False
        while not truncated:
            action = rng.integers(0, N_BINS, size=NUM_ASSETS)
            _, reward, _, truncated, _ = env.step(action)
            total += reward
        self.assertGreater(env.trades_count, 5)
        expected = np.log(env.portfolio_value / env.config.budget_initial)
        self.assertAlmostEqual(total, expected, places=6)

    def test_trades_only_fill_at_decision_bars(self):
        prices = np.full((WINDOW + 3 * K + 1, NUM_ASSETS), 100.0)
        env = _build_env(prices=prices, decision_interval=K)
        pv0 = env.portfolio_value
        _, reward, _, _, _ = env.step(np.array([10, 10]))
        self.assertEqual(env.trades_count, 2)
        # Flat prices: the whole interval's reward is the buy fee, paid once
        self.assertAlmostEqual(
            reward, np.log((pv0 - env.fees_paid_total) / pv0), places=6
        )

    def test_calmar_annualizes_by_elapsed_minutes(self):
        # The same 1-minute path sampled every K bars spans the same time
        pv = 100.0 * np.exp(np.linspace(0.0, 0.01, 30 * K + 1))
        pv[10 * K] *= 0.98  # a dip that both samplings see
        per_minute = get_calmar_from_portfolio_series(pv)
        per_decision = get_calmar_from_portfolio_series(pv[::K], minutes_per_point=K)
        self.assertAlmostEqual(per_decision / per_minute, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
