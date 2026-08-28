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
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from crypto_rl.config import RLConfig
from crypto_rl.env.metrics import calculate_calmar_ratio
from crypto_rl.env.minimal_env import MinimalCryptoEnv

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
        config: RLConfig,
        state_path: Path,
        run_id: str,
        total_timesteps: int,
        num_data_rows: int,
        training_start_str: str | None = None,
        training_end_str: str | None = None,
    ):
        super().__init__()
        self.state_path = state_path
        self.config = config
        self.window_size = config.window_size
        self.reward_type = config.reward_type
        self.run_id = run_id
        self.total_timesteps = total_timesteps
        self.num_data_rows = num_data_rows
        self.check_freq = config.check_freq
        # Always store started_at in UTC so _parse_dashboard_ts can parse it reliably.
        self.start_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.last_portfolio_value = config.budget_initial
        self.training_start_str = training_start_str
        self.training_end_str = training_end_str
        # Maximum data-points kept in memory per series to avoid unbounded RAM growth
        # during very long experiments (e.g. 400 k rows / 1.2 M timesteps).
        # The dashboard JSON is further trimmed to _MAX_JSON_POINTS at write time.
        self._MAX_SERIES_POINTS: int = 600
        self._MAX_JSON_POINTS: int = 300

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
        self.peak_portfolio_value = config.budget_initial

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
                float(portfolio_values[0]) if portfolio_values else self.config.budget_initial
            )
            episode_counts = self.training_env.get_attr("episode_count")
            current_episode = int(episode_counts[0]) if episode_counts else 1
            holdings_list = self.training_env.get_attr("holdings")
            current_holdings = holdings_list[0] if holdings_list else np.zeros(1)

            # Simplified trade count
            if np.sum(np.abs(current_holdings)) > 1e-8:
                self.current_trades += 1

            winning_trades_list = self.training_env.get_attr("winning_trades_count")
            total_closed_list = self.training_env.get_attr("total_closed_trades")
            winning_trades = int(winning_trades_list[0]) if winning_trades_list else 0
            total_closed = int(total_closed_list[0]) if total_closed_list else 0
            win_rate = (winning_trades / max(1, total_closed)) * 100.0

            # Return and Drawdown
            total_return = (current_portfolio / self.config.budget_initial - 1.0) * 100.0
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

            # Trim every series in-place to avoid unbounded RAM growth on long runs.
            cap = self._MAX_SERIES_POINTS
            for key in self.series:
                if len(self.series[key]) > cap:
                    self.series[key] = self.series[key][-cap:]
        except Exception as e:
            # Swallow metric-collection errors so training is never interrupted
            print(e)

    def _write_state(self, status: str = "running") -> None:
        try:
            # Trim to the last _MAX_JSON_POINTS entries so the JSON file stays
            # small and the browser renders charts without freezing.
            n = self._MAX_JSON_POINTS
            series_data = {key: data[-n:] for key, data in self.series.items()}

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
                    "total_timesteps": int(self.total_timesteps),
                    "progress_pct": int(
                        100
                        * min(
                            1.0, float(self.num_timesteps) / float(self.total_timesteps)
                        )
                    ),
                    "training_start": self.training_start_str,
                    "training_end": self.training_end_str,
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
        except Exception as e:
            print(e)


class TrialEvalCallback(MaskableEvalCallback):
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


# New checkpoint callback that saves model when Calmar improves
class CheckpointCallback(BaseCallback):
    """
    Save model checkpoint whenever a new best Calmar ratio is achieved on a test environment.

    Parameters
    ----------
    checkpoint_dir: Path
        Directory where checkpoint files will be saved.
    test_env: MinimalCryptoEnv
        Environment used for evaluation to compute Calmar.
    check_freq: int
        How often (in timesteps) to evaluate and possibly checkpoint.
    max_checkpoints: int
        Maximum number of checkpoint files to retain globally. Older ones are deleted.
    """

    def __init__(
        self,
        config: RLConfig,
        checkpoint_dir: Path,
        test_env: MinimalCryptoEnv | ActionMasker,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.test_env = test_env
        self.check_freq = config.check_freq
        self.max_checkpoints = config.max_checkpoints
        self.best_calmar = -float("inf")
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            calmar = self._evaluate_calmar()
            if calmar > self.best_calmar:
                self.best_calmar = calmar
                timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                ckpt_path = (
                    self.checkpoint_dir
                    / f"checkpoint_{self.num_timesteps}_calmar_{calmar:.4f}_{timestamp}.zip"
                )
                self.model.save(str(ckpt_path))
                # Enforce max checkpoints globally
                all_ckpts = sorted(
                    self.checkpoint_dir.parent.parent.glob("**/*.zip"),
                    key=lambda p: p.stat().st_mtime,
                )
                if len(all_ckpts) > self.max_checkpoints:
                    for old_ckpt in all_ckpts[: -self.max_checkpoints]:
                        try:
                            old_ckpt.unlink()
                        except Exception as e:
                            print(e)

        return True

    def _evaluate_calmar(self) -> float:
        """Run a full evaluation on the test environment and compute Calmar ratio."""
        obs, _ = self.test_env.reset()
        done = False
        # USE .unwrapped TO ACCESS BASE ENVIRONMENT ATTRIBUTES
        base_env: MinimalCryptoEnv = self.test_env.unwrapped
        portfolio_values = [{"step": 0, "value": float(base_env.portfolio_value)}]
        while not done:
            # EXTRACT MASKS AND EXPAND TO 2D FOR PREDICTION
            current_masks = np.array([self.test_env.action_masks()])
            action, _ = self.model.predict(
                obs, action_masks=current_masks, deterministic=True
            )
            obs, _, done, _, _ = self.test_env.step(action)
            portfolio_values.append(
                {
                    "step": len(portfolio_values),
                    "value": float(base_env.portfolio_value),
                }
            )
        calmar = calculate_calmar_ratio(portfolio_values)
        return calmar
