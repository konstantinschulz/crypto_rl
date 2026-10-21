"""Walk-forward metrics and the model-selection score (ROADMAP P2-3).

Each fold reports excess log return over the equal-weight basket, maxDD and
turnover; folds are summarized as mean ± spread. Checkpoints, pruning and the
Optuna objective all use ``excess log return − λ·maxDD``.
"""

import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.callbacks import UnifiedEvalCallback
from crypto_rl.config import RLConfig
from crypto_rl.env.metrics import (
    episode_metrics,
    max_drawdown,
    selection_score,
    summarize_folds,
)
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.fold_runner import _eval_env_args, _run_episode
from tests.test_baselines import DUMMY_VEC_ENV_ARGS, _ConstantTargetModel, _make_arrays

FEE = RLConfig().fee_rate


class MetricHelpersTest(unittest.TestCase):
    def test_max_drawdown(self):
        self.assertAlmostEqual(max_drawdown(np.array([100, 120, 90, 130, 65])), 0.5)
        self.assertEqual(max_drawdown(np.array([100, 101, 102])), 0.0)
        self.assertEqual(max_drawdown(np.array([100])), 0.0)

    def test_episode_metrics(self):
        m = episode_metrics(
            initial_pv=100.0,
            final_pv=110.0,
            bar_pvs=[105.0, 84.0, 111.0],
            equal_weight_final_pv=110.0,
            turnover=2.0,
            n_bars=2880,
            drawdown_coef=0.5,
        )
        self.assertAlmostEqual(m["return"], 0.10)
        self.assertAlmostEqual(m["log_return"], math.log(1.1))
        self.assertAlmostEqual(m["excess_log_return"], 0.0)
        self.assertAlmostEqual(m["max_drawdown"], 0.2)  # 105 → 84
        self.assertAlmostEqual(m["turnover_per_day"], 1.0)
        self.assertAlmostEqual(m["score"], -0.1)

    def test_score_is_monotonic_and_continuous_at_zero(self):
        # The old Calmar objective jumps at zero return; this must not
        self.assertAlmostEqual(
            selection_score(-1e-9, 0.1, 0.5), selection_score(1e-9, 0.1, 0.5)
        )
        for excess in (-0.2, 0.0, 0.2):
            self.assertLess(
                selection_score(excess, 0.1, 0.5),
                selection_score(excess + 0.01, 0.1, 0.5),
            )
            self.assertGreater(
                selection_score(excess, 0.1, 0.5),
                selection_score(excess, 0.2, 0.5),
            )

    def test_summarize_folds(self):
        summary = summarize_folds([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        self.assertEqual(
            summary["x"], {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}
        )
        self.assertEqual(summarize_folds([{"x": 4.0}])["x"]["std"], 0.0)
        self.assertEqual(summarize_folds([]), {})


def _env_args(is_fast_eval: bool, **config_kwargs):
    config = RLConfig(
        disable_logging=True,
        parquet_path=None,
        window_size=120,
        max_asset_allocation=1.0,
        **config_kwargs,
    )
    prices, static, vol = _make_arrays(3, 12000, 2)
    return _eval_env_args(
        (prices, static, vol, ["ETHUSDT", "BTCUSDT"]),
        run_id="t",
        is_fast_eval=is_fast_eval,
        config=config,
        logger=None,
    )


def _obs_rms(env_args) -> RunningMeanStd:
    obs_dim = MinimalCryptoEnv(**env_args).observation_space.shape[0]
    return RunningMeanStd(shape=(obs_dim,))


def _run(action: list[int], is_fast_eval: bool = False, **config_kwargs):
    env_args = _env_args(is_fast_eval, **config_kwargs)
    return _run_episode(
        _ConstantTargetModel(action),
        env_args,
        _obs_rms(env_args),
        DUMMY_VEC_ENV_ARGS,
        desc="test",
        show_progress=False,
    )


class EpisodeMetricsTest(unittest.TestCase):
    def test_all_in_btc(self):
        # One buy of all cash: S / (1 + fee) of notional, then a single
        # holding never drifts off target, so no further trades
        result = _run([0, 20])
        m = result["metrics"]
        self.assertAlmostEqual(m["turnover"], 1.0 / (1.0 + FEE), places=6)
        baselines = result["baselines"]
        self.assertAlmostEqual(
            m["excess_log_return"],
            math.log(baselines["btc"] / baselines["equal_weight"]),
            places=5,
        )
        self.assertGreater(m["max_drawdown"], 0.0)

    def test_never_trading(self):
        result = _run([0, 0])
        m = result["metrics"]
        self.assertEqual(m["turnover"], 0.0)
        self.assertEqual(m["max_drawdown"], 0.0)
        self.assertAlmostEqual(
            m["score"],
            math.log(result["baselines"]["cash"] / result["baselines"]["equal_weight"]),
        )

    def test_drawdown_coef_only_changes_the_score(self):
        low = _run([10, 10], objective_drawdown_coef=0.0)["metrics"]
        high = _run([10, 10], objective_drawdown_coef=2.0)["metrics"]
        self.assertAlmostEqual(low["score"], low["excess_log_return"])
        self.assertAlmostEqual(
            high["score"], high["excess_log_return"] - 2.0 * high["max_drawdown"]
        )
        self.assertEqual(low["excess_log_return"], high["excess_log_return"])


class EvalCallbackScoreTest(unittest.TestCase):
    def test_checkpoint_score_matches_final_evaluation_score(self):
        # Checkpoints are ranked by the same score the fold is reported with
        env_args = _env_args(is_fast_eval=True)
        obs_rms = _obs_rms(env_args)
        model = _ConstantTargetModel([10, 10])
        raw_env = MinimalCryptoEnv(**env_args)
        eval_env = VecNormalize(
            DummyVecEnv([lambda: Monitor(raw_env)]), **DUMMY_VEC_ENV_ARGS
        )
        eval_env.obs_rms = obs_rms
        with tempfile.TemporaryDirectory() as tmp:
            callback = UnifiedEvalCallback(
                config=dataclasses.replace(env_args["config"], eval_freq=1000),
                eval_env=eval_env,
                trial=None,
                checkpoint_dir=Path(tmp),
            )
            callback.model = model
            _, score = callback._run_evaluation()
        expected = _run_episode(
            model, env_args, obs_rms, DUMMY_VEC_ENV_ARGS, "test", False
        )["metrics"]["score"]
        self.assertAlmostEqual(score, expected, places=10)


if __name__ == "__main__":
    unittest.main()
