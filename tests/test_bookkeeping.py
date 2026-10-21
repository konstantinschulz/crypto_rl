"""Eval and checkpoint bookkeeping (ROADMAP P0-6).

Evals fire on ``n_calls`` regardless of ``n_envs``, checkpoint trimming stays
inside one fold's files, ``traded_coins`` only counts fills, and the final-test
episode reports the path of its own action log.
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from stable_baselines3.common.running_mean_std import RunningMeanStd

from crypto_rl.callbacks import UnifiedEvalCallback
from crypto_rl.config import RLConfig
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.fold_runner import _eval_env_args, _run_episode
from tests.test_baselines import DUMMY_VEC_ENV_ARGS, _ConstantTargetModel, _make_arrays


class _StubModel:
    def __init__(self, n_envs: int, num_timesteps: int = 0):
        self.env = SimpleNamespace(num_envs=n_envs)
        self.num_timesteps = num_timesteps
        self.logger = None

    def get_env(self):
        return self.env

    def save(self, path):
        Path(path).touch()


def _eval_callback(checkpoint_dir: Path, eval_freq: int, **config_kwargs):
    config = RLConfig(eval_freq=eval_freq, **config_kwargs)
    cb = UnifiedEvalCallback(
        config=config, eval_env=None, trial=None, checkpoint_dir=checkpoint_dir
    )
    cb._save_best_model = lambda: None
    return cb


class EvalCadenceTest(unittest.TestCase):
    def _count_evals(self, n_envs: int, timesteps: int, eval_freq: int) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            cb = _eval_callback(
                Path(tmp), eval_freq, timesteps=timesteps, checkpoint=False
            )
            evals = []
            cb._run_evaluation = lambda: (evals.append(1), (0.0, 0.0))[1]
            model = _StubModel(n_envs)
            cb.init_callback(model)
            while model.num_timesteps < timesteps:
                model.num_timesteps += n_envs
                cb.on_step()
            return len(evals)

    def test_eval_count_does_not_depend_on_n_envs(self):
        # Optuna setup: 1.5M steps, eval every 150k. With the old
        # `num_timesteps % eval_freq` trigger n_envs=11 gave 0 evals.
        for n_envs in (1, 7, 8, 9, 11):
            with self.subTest(n_envs=n_envs):
                self.assertEqual(self._count_evals(n_envs, 1_500_000, 150_000), 10)

    def test_resumed_model_keeps_its_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _eval_callback(Path(tmp), 100, timesteps=1000, checkpoint=False)
            cb.init_callback(_StubModel(n_envs=4, num_timesteps=400))
            self.assertEqual(cb.n_calls, 100)


class CheckpointTrimTest(unittest.TestCase):
    def test_trims_only_this_folds_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            other_fold = ckpt_dir / "fold_1_step_10_calmar_1.0000_x.zip"
            best = ckpt_dir / "best_model.zip"
            unrelated = ckpt_dir / "chunk_model.zip"
            for f in (other_fold, best, unrelated):
                f.touch()
            cb = _eval_callback(ckpt_dir, 100, max_checkpoints=2)
            cb.init_callback(_StubModel(n_envs=1))
            for step in range(1, 5):
                cb.num_timesteps = step
                cb._save_checkpoint(float(step))
                # Distinct mtimes so trimming order is deterministic
                newest = max(ckpt_dir.glob("fold_0_*.zip"), key=os.path.getctime)
                os.utime(newest, (step, step))
            remaining = sorted(p.name for p in ckpt_dir.glob("fold_0_*.zip"))
            self.assertEqual(len(remaining), 2)
            self.assertTrue(all("_step_3_" in n or "_step_4_" in n for n in remaining))
            for f in (other_fold, best, unrelated):
                self.assertTrue(f.exists(), f.name)


class EpisodeBookkeepingTest(unittest.TestCase):
    def setUp(self):
        # init_log writes under ./logs/<run_id>; keep it out of the repo.
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _run(self, action: list[int]):
        config = RLConfig(
            disable_logging=False,
            parquet_path=None,
            window_size=120,
            max_asset_allocation=1.0,
        )
        prices, static, vol = _make_arrays(3, 3000, 2)
        env_args = _eval_env_args(
            (prices, static, vol, ["ETHUSDT", "BTCUSDT"]),
            run_id="t",
            is_fast_eval=False,
            config=config,
            logger=None,
        )
        obs_dim = MinimalCryptoEnv(**env_args).observation_space.shape[0]
        return _run_episode(
            _ConstantTargetModel(action),
            env_args,
            RunningMeanStd(shape=(obs_dim,)),
            DUMMY_VEC_ENV_ARGS,
            desc="test",
            show_progress=False,
        )

    def test_traded_coins_counts_only_fills(self):
        # All cash every step: nothing was traded.
        self.assertEqual(self._run([0, 0])["traded_coins"], [])
        # 100% BTC every step: only BTC was traded, though ETH was in every action.
        self.assertEqual(self._run([0, 20])["traded_coins"], ["BTCUSDT"])

    def test_fill_counts_survive_the_auto_reset(self):
        # The VecEnv resets the env on the terminal step; the reported counts
        # must still be the finished episode's.
        self.assertEqual(self._run([0, 20])["fill_counts"], {"BUY": 1})

    def test_reports_its_own_action_log(self):
        result = self._run([0, 20])
        log_path = Path(result["action_log_path"])
        self.assertTrue(log_path.exists())
        self.assertTrue(log_path.name.startswith("actions_eval_ep1_"))
        df = pd.read_parquet(log_path)
        self.assertEqual(len(df), result["steps"])
        self.assertAlmostEqual(
            df["portfolio"].iloc[0], result["initial_portfolio_value"]
        )


if __name__ == "__main__":
    unittest.main()
