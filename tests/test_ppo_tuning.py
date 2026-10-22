"""PPO re-tune (ROADMAP P2-4): defaults, entropy schedule and Optuna search space."""

import sys
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from crypto_rl.callbacks import EntropyDecayCallback
from crypto_rl.config import RLConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import optuna_search


def _ent_coef_at(callback: EntropyDecayCallback, num_timesteps: int) -> float:
    callback.model = SimpleNamespace(num_timesteps=num_timesteps, ent_coef=None)
    callback.on_step()
    return callback.model.ent_coef


def test_defaults_are_the_p2_4_starting_point():
    cfg = RLConfig()
    assert cfg.learning_rate == 3e-4
    assert cfg.clip_range == 0.2
    assert cfg.ent_coef_initial == cfg.ent_coef_final == 0.01


def test_equal_initial_and_final_ent_coef_is_constant():
    cb = EntropyDecayCallback(0.01, 0.01, total_timesteps=1000)
    assert [_ent_coef_at(cb, t) for t in (0, 500, 1000, 2000)] == [0.01] * 4


def test_ent_coef_decays_linearly_to_the_end_of_the_run():
    cb = EntropyDecayCallback(0.01, 0.002, total_timesteps=1000)
    assert _ent_coef_at(cb, 0) == pytest.approx(0.01)
    assert _ent_coef_at(cb, 500) == pytest.approx(0.006)
    # Still above the floor at 90%, unlike the old timesteps·(1 − test_fraction) horizon
    assert _ent_coef_at(cb, 900) > 0.002
    assert _ent_coef_at(cb, 1000) == pytest.approx(0.002)
    assert _ent_coef_at(cb, 5000) == pytest.approx(0.002)


def test_optuna_searches_only_the_narrowed_space():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    cfg = optuna_search.get_optuna_config(trial)
    assert set(trial.params) == {
        "learning_rate",
        "ent_coef",
        "gamma",
        "drawdown_penalty_coef",
    }
    assert cfg.ent_coef_initial == cfg.ent_coef_final == trial.params["ent_coef"]
    assert cfg.clip_range == 0.2
    assert cfg.n_envs == 9


def test_optuna_baseline_trial_matches_the_defaults():
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(optuna_search.BASELINE_PARAMS)
    cfg = optuna_search.get_optuna_config(study.ask())
    default = RLConfig()
    assert cfg.learning_rate == default.learning_rate
    assert cfg.clip_range == default.clip_range
    assert cfg.ent_coef_initial == default.ent_coef_initial
    assert cfg.ent_coef_final == default.ent_coef_final
    assert cfg.gamma == default.gamma
    assert cfg.drawdown_penalty_coef == default.drawdown_penalty_coef


def test_resumed_trial_reuses_frozen_params():
    study = optuna.create_study(direction="maximize")
    frozen = dict(optuna_search.BASELINE_PARAMS, learning_rate=1e-4, ent_coef=0.003)
    cfg = optuna_search.get_optuna_config(study.ask(), override_params=frozen)
    assert cfg.learning_rate == 1e-4
    assert cfg.ent_coef_initial == cfg.ent_coef_final == 0.003
