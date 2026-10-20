"""Baselines over the exact window the bot is scored on (ROADMAP P0-5).

Equal-weight buy-and-hold, BTC buy-and-hold and all-cash must cover the same
steps as the bot's evaluation episode and pay the same fees.
"""

import unittest

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from crypto_rl.config import RLConfig
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.evaluation_utils import (
    compute_window_baselines,
    excess_returns,
)
from crypto_rl.experiment.fold_runner import _eval_env_args, _run_episode

DUMMY_VEC_ENV_ARGS = {
    "norm_reward": False,
    "norm_obs": True,
    "clip_obs": 10.0,
    "training": False,
}


def _make_arrays(seed: int, n_bars: int, n_assets: int):
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.01, size=(n_bars + 1, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    static_dim = MACRO_DIM + STATIC_PER_ASSET_DIM * n_assets
    static = rng.normal(size=(n_bars + 1, static_dim))
    norm_vol = np.abs(log_rets) + 1e-3
    return prices, static, norm_vol


class _ScriptedModel:
    """Returns `first_action` on the first call and HOLD afterwards."""

    def __init__(self, first_action: list[int]):
        self.first_action = np.array([first_action])
        self.calls = 0

    def predict(self, obs, action_masks=None, deterministic=True):
        self.calls += 1
        if self.calls == 1:
            return self.first_action, None
        return np.zeros_like(self.first_action), None


class ComputeWindowBaselinesTest(unittest.TestCase):
    def test_values_use_window_endpoints_and_fees(self):
        prices = np.array(
            [
                [999.0, 999.0],  # before the window: ignored
                [100.0, 10.0],  # entry: prices[start_step - 1]
                [150.0, 50.0],  # inside the window: ignored
                [200.0, 5.0],  # exit: prices[max_steps - 1]
                [999.0, 999.0],  # never reached by the episode: ignored
            ]
        )
        fee = 0.001
        b = compute_window_baselines(
            prices,
            start_step=2,
            max_steps=4,
            asset_names=["ETHUSDT", "BTCUSDT"],
            fee_rate=fee,
            budget_initial=1000.0,
        )
        keep = (1 - fee) / (1 + fee)  # one buy fee and one sell fee
        self.assertAlmostEqual(b["btc"], 1000.0 * 0.5 * keep)
        self.assertAlmostEqual(b["equal_weight"], 500.0 * (2.0 + 0.5) * keep)
        self.assertEqual(b["cash"], 1000.0)

        ex = excess_returns(1100.0, b, 1000.0)
        self.assertAlmostEqual(ex["cash"], 0.1)
        self.assertAlmostEqual(ex["btc"], (1100.0 - b["btc"]) / 1000.0)

    def test_falls_back_to_first_asset_without_btc(self):
        prices = np.array([[100.0, 10.0], [110.0, 10.0]])
        b = compute_window_baselines(
            prices, 1, 2, ["ETHUSDT", "SOLUSDT"], fee_rate=0.0, budget_initial=100.0
        )
        self.assertAlmostEqual(b["btc"], 110.0)


class RunEpisodeBaselinesTest(unittest.TestCase):
    def _run(self, first_action: list[int], is_fast_eval: bool):
        config = RLConfig(
            disable_logging=True,
            parquet_path=None,
            window_size=120,
            max_asset_allocation=1.0,
            max_single_step_allocation=1.0,
        )
        prices, static, vol = _make_arrays(3, 12000, 2)
        names = ["ETHUSDT", "BTCUSDT"]
        env_args = _eval_env_args(
            (prices, static, vol, names),
            run_id="t",
            is_fast_eval=is_fast_eval,
            config=config,
            logger=None,
        )
        obs_dim = MinimalCryptoEnv(**env_args).observation_space.shape[0]
        return _run_episode(
            _ScriptedModel(first_action),
            env_args,
            RunningMeanStd(shape=(obs_dim,)),
            DUMMY_VEC_ENV_ARGS,
            desc="test",
            show_progress=False,
        )

    def test_all_in_btc_on_first_step_matches_btc_baseline(self):
        # BUY asset 1 (BTCUSDT) with 100% of cash, then hold to the end.
        # Both the full and the fast (offset) window must line up exactly.
        for fast in (False, True):
            result = self._run([1, 1, 100], is_fast_eval=fast)
            self.assertAlmostEqual(
                result["final_portfolio_value"] / result["baselines"]["btc"],
                1.0,
                places=5,
            )
            self.assertAlmostEqual(result["excess_returns"]["btc"], 0.0, places=5)
            self.assertNotAlmostEqual(
                result["baselines"]["btc"], result["baselines"]["equal_weight"]
            )

    def test_never_trading_matches_cash_baseline(self):
        result = self._run([0, 0, 0], is_fast_eval=False)
        self.assertEqual(result["final_portfolio_value"], result["baselines"]["cash"])
        self.assertEqual(result["excess_returns"]["cash"], 0.0)


if __name__ == "__main__":
    unittest.main()
