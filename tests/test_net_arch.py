"""The PPO critic must have hidden layers.

SB3 reads the value network from the ``"vf"`` key of ``net_arch``. A ``"qf"``
key is ignored silently, which leaves the critic as one linear layer.
"""

import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_rl.config import RLConfig
from crypto_rl.experiment.fold_runner import build_policy_kwargs


class _TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([6, 6])

    def reset(self, seed=None, options=None):
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(8, dtype=np.float32), 0.0, False, False, {}


def _linear_layers(module: torch.nn.Module) -> list[torch.nn.Linear]:
    return [m for m in module.modules() if isinstance(m, torch.nn.Linear)]


class NetArchTest(unittest.TestCase):
    def test_policy_and_value_networks_have_hidden_layers(self):
        config = RLConfig(net_arch_dim=32)
        kwargs = build_policy_kwargs(config)
        self.assertEqual(set(kwargs["net_arch"]), {"pi", "vf"})

        model = PPO(
            "MlpPolicy",
            DummyVecEnv([_TinyEnv]),
            policy_kwargs=kwargs,
            device="cpu",
            verbose=0,
            n_steps=8,
            batch_size=8,
        )
        mlp = model.policy.mlp_extractor
        value_layers = _linear_layers(mlp.value_net)
        policy_layers = _linear_layers(mlp.policy_net)

        # Two hidden layers each, so the critic is not a single linear map.
        self.assertEqual(len(value_layers), 2)
        self.assertEqual(len(policy_layers), 2)
        self.assertEqual(value_layers[0].out_features, 32)
        self.assertEqual(value_layers[1].out_features, 32)


if __name__ == "__main__":
    unittest.main()
