"""Train / validation / test separation.

Validation drives intermediate evals, checkpoint ranking and the Optuna
objective. The test slice is only used for the final report, over the whole
test fold, with the best-on-validation checkpoint.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.callbacks import UnifiedEvalCallback
from crypto_rl.config import RLConfig
from crypto_rl.data import FoldSplit, make_walk_forward_splits
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.data_utils import (
    assert_chronological_bounds,
    fold_cache_prefixes,
    split_cache_name,
)
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.fold_runner import _eval_env_args, _run_episode

# Microsecond timestamps, one per minute
TIMES = 1_700_000_000_000_000 + np.arange(10_000) * 60_000_000
DUMMY_VEC_ENV_ARGS = {
    "norm_reward": False,
    "norm_obs": True,
    "clip_obs": 10.0,
    "training": False,
}


def _bounds(split: FoldSplit) -> dict[str, tuple[int, int]]:
    return {
        "train": (int(TIMES[0]), split.t_train_max),
        "val": (split.t_val_min, split.t_val_max),
        "test": (split.t_test_min, split.t_test_max),
    }


class SplitLayoutTest(unittest.TestCase):
    def _check_fold(self, split: FoldSplit):
        b = _bounds(split)
        assert_chronological_bounds(b)  # disjoint, ordered, distinct starts
        # Contiguous: val directly follows train, test directly follows val
        i_train = np.searchsorted(TIMES, split.t_train_max)
        i_val0 = np.searchsorted(TIMES, split.t_val_min)
        i_val1 = np.searchsorted(TIMES, split.t_val_max)
        i_test0 = np.searchsorted(TIMES, split.t_test_min)
        self.assertEqual(i_val0, i_train + 1)
        self.assertEqual(i_test0, i_val1 + 1)

    def test_single_fold_layout_and_sizes(self):
        (split,) = make_walk_forward_splits(TIMES, 1, 0.2, 0.1)
        self._check_fold(split)
        b = _bounds(split)
        n = lambda name: np.searchsorted(TIMES, b[name][1]) - np.searchsorted(
            TIMES, b[name][0]
        ) + 1
        self.assertEqual(n("test"), 2000)
        self.assertEqual(n("val"), 1000)
        self.assertEqual(n("train"), 7000)
        self.assertEqual(split.t_test_max, int(TIMES[-1]))

    def test_multi_fold_layout(self):
        splits = make_walk_forward_splits(TIMES, 3, 0.2, 0.1)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self._check_fold(split)
        # Expanding window, test slices tile the tail without overlap
        for a, b in zip(splits, splits[1:]):
            self.assertLess(a.t_train_max, b.t_train_max)
            self.assertLess(a.t_test_max, b.t_test_min)
        self.assertEqual(splits[-1].t_test_max, int(TIMES[-1]))

    def test_test_slices_do_not_depend_on_val_fraction(self):
        for folds in (1, 3):
            small = make_walk_forward_splits(TIMES, folds, 0.2, 0.05)
            large = make_walk_forward_splits(TIMES, folds, 0.2, 0.15)
            self.assertEqual(
                [(s.t_test_min, s.t_test_max) for s in small],
                [(s.t_test_min, s.t_test_max) for s in large],
            )

    def test_invalid_fractions_raise(self):
        with self.assertRaises(ValueError):
            make_walk_forward_splits(TIMES, 1, 0.2, 0.0)  # empty val
        with self.assertRaises(ValueError):
            make_walk_forward_splits(TIMES, 1, 0.0, 0.1)  # empty test
        with self.assertRaises(ValueError):
            make_walk_forward_splits(TIMES, 1, 0.2, 0.9)  # no training data
        with self.assertRaises(ValueError):
            make_walk_forward_splits(TIMES, 3, 0.2, 0.3)  # val exceeds first fold

    def test_overlapping_or_identical_slices_are_rejected(self):
        with self.assertRaises(AssertionError):
            assert_chronological_bounds({"train": (0, 10), "val": (10, 20)})
        with self.assertRaises(AssertionError):
            assert_chronological_bounds({"train": (0, 10), "val": (0, 5)})

    def test_cache_prefixes_are_distinct_per_slice_and_per_val_fraction(self):
        split = make_walk_forward_splits(TIMES, 1, 0.2, 0.1)[0]
        data_range = (int(TIMES[0]), int(TIMES[-1]))
        config = RLConfig()
        p = fold_cache_prefixes(config, 9, 0, data_range, split)
        self.assertEqual(len({p.train, p.val, p.test}), 3)
        other = fold_cache_prefixes(
            RLConfig(val_fraction=0.2), 9, 0, data_range, split
        )
        self.assertTrue(set(p).isdisjoint(set(other)))
        self.assertNotEqual(
            split_cache_name(config, data_range),
            split_cache_name(RLConfig(val_fraction=0.2), data_range),
        )


class _FakeModel:
    num_timesteps = 0

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"model")


class BestModelCallbackTest(unittest.TestCase):
    def test_best_model_is_kept_and_trimming_is_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            own = root / "run_a" / "checkpoints_fold_1"
            other = root / "run_b" / "checkpoints_fold_1"
            other.mkdir(parents=True)
            foreign = other / "fold_0_step_1_calmar_1.0000_x.zip"
            foreign.write_bytes(b"other run")

            obs_rms = SimpleNamespace(mean=np.zeros(3), var=np.ones(3), count=1.0)
            callback = UnifiedEvalCallback(
                config=RLConfig(max_checkpoints=2, eval_freq=1000),
                eval_env=SimpleNamespace(obs_rms=obs_rms),
                trial=None,
                checkpoint_dir=own,
            )
            callback.model = _FakeModel()

            callback._save_best_model()
            obs_rms.mean[:] = 99.0  # keeps drifting during training
            self.assertEqual(callback.best_model_path, own / "best_model.zip")
            self.assertTrue(callback.best_model_path.exists())
            np.testing.assert_array_equal(callback.best_obs_rms.mean, np.zeros(3))

            for step in range(1, 5):
                callback.num_timesteps = step
                callback._save_checkpoint(float(step))

            ranked = sorted(own.glob("fold_*_step_*.zip"))
            self.assertEqual(len(ranked), 2)  # trimmed to max_checkpoints
            self.assertTrue((own / "best_model.zip").exists())
            self.assertTrue(foreign.exists(), "must not delete other runs' files")


def _make_arrays(seed: int, n_bars: int, n_assets: int):
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.01, size=(n_bars + 1, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    static_dim = MACRO_DIM + STATIC_PER_ASSET_DIM * n_assets
    static = rng.normal(size=(n_bars + 1, static_dim))
    norm_vol = np.abs(log_rets) + 1e-3
    return prices, static, norm_vol, [f"A{i}" for i in range(n_assets)]


class RunEpisodeTest(unittest.TestCase):
    def test_final_evaluation_covers_the_whole_fold(self):
        n_assets = 2
        config = RLConfig(
            disable_logging=True, parquet_path=None, n_envs=1, window_size=120
        )
        train = _make_arrays(0, 1500, n_assets)
        # Longer than one episode (10080), so fast eval starts 25% in
        val = _make_arrays(1, 12000, n_assets)

        def make_train_env():
            prices, static, vol, names = train
            env = MinimalCryptoEnv(
                prices_arr=prices,
                static_obs=static,
                norm_vol_arr=vol,
                asset_names=names,
                config=config,
            )
            return ActionMasker(env, get_action_mask)

        train_env = VecNormalize(DummyVecEnv([make_train_env]), norm_reward=True)
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=64,
            batch_size=64,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
        )
        model.learn(total_timesteps=128)

        results = {}
        for fast in (True, False):
            results[fast] = _run_episode(
                model,
                _eval_env_args(
                    val,
                    run_id="t",
                    is_fast_eval=fast,
                    config=config,
                    logger=None,
                ),
                train_env.obs_rms,
                DUMMY_VEC_ENV_ARGS,
                desc="test",
                show_progress=False,
            )

        # The full episode covers the whole fold; the fast one is a shorter slice
        self.assertEqual(results[False]["steps"], 12000 - config.window_size)
        self.assertLess(results[True]["steps"], results[False]["steps"])
        self.assertLessEqual(results[True]["steps"], config.eval_length)
        for key in ("calmar", "per_asset_stats", "final_portfolio_value"):
            self.assertIn(key, results[False])


if __name__ == "__main__":
    unittest.main()
