"""Episode boundary (ROADMAP P1-2).

The time limit is a truncation, not a termination. Open positions are not
force-liquidated: the reward stays mark-to-market, and the reported final PV
charges the exit fee without executing or counting a sell.
"""

import unittest

import numpy as np

from tests.test_reward import HOLD, NUM_ASSETS, WINDOW, _build_env


def _run_to_end(env, actions):
    """Step through ``actions`` and then HOLD until the episode ends."""
    actions = list(actions)
    while True:
        action = np.array(actions.pop(0) if actions else HOLD)
        out = env.step(action)
        if out[2] or out[3]:
            return out


class EpisodeBoundaryTest(unittest.TestCase):
    def test_boundary_is_truncation_not_termination(self):
        env = _build_env()
        terminated = truncated = False
        n_steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(np.array(HOLD))
            n_steps += 1
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.current_step, env.max_steps)
        self.assertGreater(n_steps, 1)

    def test_open_positions_are_not_liquidated(self):
        env = _build_env()
        _, _, _, _, info = _run_to_end(env, [[1, 0, 50], [1, 1, 50]])
        # Holdings survive the boundary, and no closing sells were recorded
        self.assertTrue(np.all(env.holdings > 0))
        self.assertEqual(env.trades_count, 2)
        self.assertEqual(info["final_trades_count"], 2)
        self.assertEqual(env.total_closed_trades, 0)
        self.assertEqual(int(env.per_asset_trades.sum()), 0)
        for stats in info["per_asset_stats"].values():
            self.assertEqual(stats["trades"], 0)
            self.assertGreater(stats["current_holdings"], 0.0)

    def test_final_pv_charges_exit_fee_but_reward_is_mark_to_market(self):
        prices = np.full((WINDOW + 10, NUM_ASSETS), 100.0)
        env = _build_env(prices=prices)
        pv0 = env.portfolio_value
        _, reward, _, _, info = _run_to_end(env, [[1, 0, 100]])
        buy_fee = env.fees_paid_total
        position_value = env.holdings[0] * 100.0
        exit_fee = position_value * env.fee_rate

        # Flat prices: the last step's reward is 0, there is no exit-fee hit
        self.assertEqual(reward, 0.0)
        # Holdings are float32, hence the loose tolerances
        self.assertAlmostEqual(env.portfolio_value, pv0 - buy_fee, places=4)
        self.assertAlmostEqual(info["final_exit_fees"], exit_fee, places=6)
        self.assertAlmostEqual(
            info["final_portfolio_value"], pv0 - buy_fee - exit_fee, places=4
        )
        self.assertAlmostEqual(info["final_fees_paid"], buy_fee + exit_fee, places=6)

    def test_per_asset_pnl_sums_to_final_pnl(self):
        env = _build_env()
        rng = np.random.default_rng(3)
        actions = [
            [rng.integers(0, 3), rng.integers(0, NUM_ASSETS), rng.integers(0, 101)]
            for _ in range(300)
        ]
        _, _, _, _, info = _run_to_end(env, actions)
        self.assertGreater(env.total_closed_trades, 0)
        total_pnl = sum(s["total_pnl"] for s in info["per_asset_stats"].values())
        self.assertAlmostEqual(
            total_pnl,
            info["final_portfolio_value"] - env.config.budget_initial,
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
