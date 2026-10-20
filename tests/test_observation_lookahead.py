"""Guards against one-bar lookahead in build_observation.

At env.current_step == t the agent may only see bars up to t-1. The action
taken from that observation fills at prices[t-1] and is rewarded on the move
prices[t-1] -> prices[t]. No observation feature may track that move.
"""

import unittest

import numpy as np

from crypto_rl.config import RLConfig
from crypto_rl.env.feature_utils import STATIC_PER_ASSET_DIM
from crypto_rl.env.minimal_env import MinimalCryptoEnv

NUM_BARS = 2500
NUM_ASSETS = 2
SEED = 0


def _build_env(seed: int) -> tuple[MinimalCryptoEnv, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.01, size=(NUM_BARS + 1, NUM_ASSETS))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    # Static features written so that a builder reading the wrong row would
    # see the bar-k return at row k: row t would then equal the rewarded move.
    macro_dim = 5
    static_dim = macro_dim + STATIC_PER_ASSET_DIM * NUM_ASSETS
    static = np.zeros((NUM_BARS + 1, static_dim), dtype=np.float64)
    static[:, :] = np.repeat(log_rets[:, :1], static_dim, axis=1)
    norm_vol = np.abs(log_rets) + 1e-3

    # parquet_path=None keeps reset() from reloading the real dataset
    config = RLConfig(disable_logging=True, parquet_path=None)
    env = MinimalCryptoEnv(
        prices_arr=prices,
        static_obs=static,
        norm_vol_arr=norm_vol,
        asset_names=[f"A{i}" for i in range(NUM_ASSETS)],
        config=config,
        is_eval=True,
    )
    return env, prices


class NoObservationLookaheadTest(unittest.TestCase):
    def test_no_observation_feature_tracks_rewarded_return(self):
        env, prices = _build_env(SEED)
        rng = np.random.default_rng(SEED + 1)
        env.reset(seed=SEED)

        obs_rows, ret_rows = [], []
        done = False
        while not done:
            c = env.current_step
            obs_rows.append(np.array(env._get_obs(), dtype=np.float64))
            # Market move the step is rewarded on: prices[c-1] -> prices[c]
            ret_rows.append(prices[c] / prices[c - 1] - 1.0)
            action = np.array(
                [rng.integers(0, 3), rng.integers(0, NUM_ASSETS), rng.integers(0, 101)]
            )
            _, _, done, _, _ = env.step(action)

        obs = np.vstack(obs_rows)
        rets = np.vstack(ret_rows)
        self.assertGreater(len(obs), 1000)

        # Holdings must exist for the portfolio features to be informative,
        # otherwise the test would pass trivially.
        self.assertTrue(np.any(env.holdings > 0) or env.per_asset_trades.sum() > 0)

        # Noise floor for n independent samples is about 1/sqrt(n).
        threshold = 5.0 / np.sqrt(len(obs))
        obs_c = obs - obs.mean(axis=0)
        ret_c = rets - rets.mean(axis=0)
        denom = np.outer(
            np.sqrt((obs_c**2).sum(axis=0)) + 1e-12,
            np.sqrt((ret_c**2).sum(axis=0)) + 1e-12,
        )
        corr = (obs_c.T @ ret_c) / denom  # shape (features, assets)
        worst = np.nanmax(np.abs(corr))
        self.assertLess(
            worst,
            threshold,
            msg=f"observation feature correlates with rewarded return: |r|={worst:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
