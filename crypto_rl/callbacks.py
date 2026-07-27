"""
crypto_rl.callbacks
===================
Stable-Baselines3 training callbacks for the crypto RL agent.

:class:`DashboardCallback` periodically writes a ``state.json`` file that
the Streamlit dashboard can poll for live metrics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from stable_baselines3.common.callbacks import EvalCallback

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover

    class BaseCallback:  # type: ignore
        """Fallback BaseCallback with minimal interface used in this script."""

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            # Return a dummy callable for any attribute used in the code
            return lambda *a, **k: None


from .env import BUDGET_INITIAL, MinimalCryptoEnv


class DashboardCallback(BaseCallback):
    """Write periodic ``state.json`` snapshots for the Streamlit dashboard.

    Parameters
    ----------
    state_path:
        Path where the JSON state file will be written on every checkpoint.
    window_size:
        Observation window size (passed through to the state file for
        display purposes).
    reward_type:
        Reward function in use (passed through to the state file).
    run_id:
        Unique identifier for this training run.
    total_timesteps:
        Total number of training timesteps (used to compute progress %).
    num_data_rows:
        Number of data rows loaded (passed through for display).
    check_freq:
        How often (in environment steps) to write the state file.
    """

    def __init__(
        self,
        state_path: Path,
        window_size: int,
        reward_type: str,
        run_id: str,
        total_timesteps: int,
        num_data_rows: int,
        check_freq: int = 500,
    ):
        super().__init__()
        self.state_path = state_path
        self.window_size = window_size
        self.reward_type = reward_type
        self.run_id = run_id
        self.total_timesteps = total_timesteps
        self.num_data_rows = num_data_rows
        self.check_freq = check_freq
        self.start_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.last_portfolio_value = BUDGET_INITIAL

        # Series data
        self.series: dict[str, list] = {
            "train_reward": [],
            "portfolio_value": [],
            "trades": [],
            "win_rate": [],
            "train_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "actor_loss": [],
            "critic_loss": [],
            "ent_coef_loss": [],
            "ent_coef": [],
            "ram_mb": [],
            "total_return_pct": [],
            "drawdown_pct": [],
        }

        self.current_trades = 0
        self.winning_trades = 0
        self.peak_portfolio_value = BUDGET_INITIAL

        try:
            import psutil

            self.psutil: Optional[object] = psutil
        except ImportError:
            self.psutil = None

    # ------------------------------------------------------------------
    # BaseCallback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._write_state(status="initializing")

    def _on_step(self) -> bool:
        # Periodically write state with current num_timesteps and real metrics
        if self.num_timesteps % self.check_freq == 0:
            self._collect_metrics()
            self._write_state(status="running")
        return True

    def _on_training_end(self) -> None:
        self._collect_metrics()
        self._write_state(status="finished")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_metrics(self) -> None:
        """Extract real portfolio value and reward from the training environment."""
        step = int(self.num_timesteps)
        try:
            # Use ep_info_buffer for smooth finalized episode metrics
            mean_ep_rew = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_ep_rew = float(
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                )
            # Always get the real current portfolio value from the environment
            portfolio_values = self.training_env.get_attr("portfolio_value")
            current_portfolio = (
                float(portfolio_values[0]) if portfolio_values else BUDGET_INITIAL
            )
            episode_counts = self.training_env.get_attr("episode_count")
            current_episode = int(episode_counts[0]) if episode_counts else 1
            holdings_list = self.training_env.get_attr("holdings")
            current_holdings = holdings_list[0] if holdings_list else np.zeros(1)

            # Simplified trade count
            if np.sum(np.abs(current_holdings)) > 1e-8:
                self.current_trades += 1

            # Win rate placeholder: assume a "win" if mean_ep_rew > 0
            if mean_ep_rew > 0:
                self.winning_trades += 1

            win_rate = (self.winning_trades / max(1, self.current_trades)) * 100.0

            # Return and Drawdown
            total_return = (current_portfolio / BUDGET_INITIAL - 1.0) * 100.0
            self.peak_portfolio_value = max(
                self.peak_portfolio_value, current_portfolio
            )
            drawdown = (1.0 - current_portfolio / self.peak_portfolio_value) * 100.0

            # Append to series
            self.series["portfolio_value"].append(
                {"step": step, "value": current_portfolio, "episode": current_episode}
            )
            self.series["train_reward"].append({"step": step, "value": mean_ep_rew})
            self.series["trades"].append(
                {"step": step, "value": int(self.current_trades)}
            )
            self.series["win_rate"].append({"step": step, "value": float(win_rate)})
            self.series["total_return_pct"].append(
                {"step": step, "value": float(total_return)}
            )
            self.series["drawdown_pct"].append({"step": step, "value": float(drawdown)})

            # Technical metrics from SB3 logger
            # SB3 uses '/' as separator, e.g., 'train/loss'
            logger_map = self.model.logger.name_to_value
            self.series["train_loss"].append(
                {"step": step, "value": float(logger_map.get("train/loss", 0.0))}
            )
            self.series["policy_loss"].append(
                {
                    "step": step,
                    "value": float(logger_map.get("train/policy_gradient_loss", 0.0)),
                }
            )
            self.series["value_loss"].append(
                {"step": step, "value": float(logger_map.get("train/value_loss", 0.0))}
            )
            self.series["approx_kl"].append(
                {"step": step, "value": float(logger_map.get("train/approx_kl", 0.0))}
            )
            self.series["clip_fraction"].append(
                {
                    "step": step,
                    "value": float(logger_map.get("train/clip_fraction", 0.0)),
                }
            )

            # SAC-specific metrics from SB3 logger
            self.series["actor_loss"].append(
                {"step": step, "value": float(logger_map.get("train/actor_loss", 0.0))}
            )
            self.series["critic_loss"].append(
                {"step": step, "value": float(logger_map.get("train/critic_loss", 0.0))}
            )
            self.series["ent_coef_loss"].append(
                {
                    "step": step,
                    "value": float(logger_map.get("train/ent_coef_loss", 0.0)),
                }
            )
            self.series["ent_coef"].append(
                {"step": step, "value": float(logger_map.get("train/ent_coef", 0.0))}
            )

            # Memory usage
            if self.psutil:
                ram = self.psutil.Process().memory_info().rss / (1024 * 1024)
                self.series["ram_mb"].append({"step": step, "value": float(ram)})

            self.last_portfolio_value = current_portfolio
        except Exception:
            # Swallow metric-collection errors so training is never interrupted
            pass

    def _write_state(self, status: str = "running") -> None:
        try:
            # Keep only the last 100 entries to avoid unbounded JSON growth
            series_data = {key: data[-100:] for key, data in self.series.items()}

            state = {
                "run": {
                    "run_id": self.run_id,
                    "mode": "minimal",
                    "status": status,
                    "started_at": self.start_ts,
                    "finished_at": (
                        None
                        if status != "finished"
                        else datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                    ),
                    "current_step": int(self.num_timesteps),
                    "progress_pct": int(
                        100
                        * min(
                            1.0, float(self.num_timesteps) / float(self.total_timesteps)
                        )
                    ),
                },
                "technical": {
                    "loss": {
                        "train": (
                            self.series["train_loss"][-1]["value"]
                            if self.series["train_loss"]
                            else None
                        )
                    },
                    "num_data_rows": self.num_data_rows,
                    "window_size": self.window_size,
                    "reward_type": self.reward_type,
                },
                "series": series_data,
                "finance": {},
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass


class TrialEvalCallback(EvalCallback):
    """Callback for evaluating a trial and reporting to Optuna, with pruning support."""

    def __init__(
        self,
        eval_env: MinimalCryptoEnv,
        trial: optuna.trial.Trial,
        n_eval_episodes: int = 1,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        # Run evaluation at the specified frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # Report mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune if Optuna decides
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
