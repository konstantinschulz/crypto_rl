"""
crypto_rl.callbacks
===================
Stable-Baselines3 training callbacks for the crypto RL agent.

:class:`DashboardCallback` periodically writes a ``state.json`` file that
the Streamlit dashboard can poll for live metrics.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna

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
            return lambda *a, **k: None


class DashboardCallback(BaseCallback):
    """Write periodic ``state.json`` snapshots for the Streamlit dashboard."""

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
        self.check_freq = config.dashboard_freq
        self.start_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.last_portfolio_value = config.budget_initial
        self.training_start_str = training_start_str
        self.training_end_str = training_end_str
        self._MAX_SERIES_POINTS: int = 600
        self._MAX_JSON_POINTS: int = 300

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

    def _on_training_start(self) -> None:
        self._write_state(status="initializing")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            self._collect_metrics()
            self._write_state(status="running")
        return True

    def _on_training_end(self) -> None:
        self._collect_metrics()
        self._write_state(status="finished")

    def _collect_metrics(self) -> None:
        step = int(self.num_timesteps)
        try:
            mean_ep_rew = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_ep_rew = float(
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                )
            portfolio_values = self.training_env.get_attr("portfolio_value")
            current_portfolio = (
                float(portfolio_values[0])
                if portfolio_values
                else self.config.budget_initial
            )
            episode_counts = self.training_env.get_attr("episode_count")
            current_episode = int(episode_counts[0]) if episode_counts else 1
            holdings_list = self.training_env.get_attr("holdings")
            current_holdings = holdings_list[0] if holdings_list else np.zeros(1)

            if np.sum(np.abs(current_holdings)) > 1e-8:
                self.current_trades += 1

            winning_trades_list = self.training_env.get_attr("winning_trades_count")
            total_closed_list = self.training_env.get_attr("total_closed_trades")
            winning_trades = int(winning_trades_list[0]) if winning_trades_list else 0
            total_closed = int(total_closed_list[0]) if total_closed_list else 0
            win_rate = (winning_trades / max(1, total_closed)) * 100.0

            total_return = (
                current_portfolio / self.config.budget_initial - 1.0
            ) * 100.0
            self.peak_portfolio_value = max(
                self.peak_portfolio_value, current_portfolio
            )
            drawdown = (1.0 - current_portfolio / self.peak_portfolio_value) * 100.0

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

            if self.psutil:
                ram = self.psutil.Process().memory_info().rss / (1024 * 1024)
                self.series["ram_mb"].append({"step": step, "value": float(ram)})

            self.last_portfolio_value = current_portfolio

            cap = self._MAX_SERIES_POINTS
            for key in self.series:
                if len(self.series[key]) > cap:
                    self.series[key] = self.series[key][-cap:]
        except Exception as e:
            print(e)

    def _write_state(self, status: str = "running") -> None:
        try:
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


class EntropyDecayCallback(BaseCallback):
    """Linearly decays PPO ent_coef from initial_ent to final_ent over total_timesteps."""

    def __init__(
        self,
        ent_coef_initial: float,
        ent_coef_final: float,
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ent_coef_initial = ent_coef_initial
        self.ent_coef_final = ent_coef_final
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / float(self.total_timesteps))
        current_ent = self.ent_coef_initial + progress * (
            self.ent_coef_final - self.ent_coef_initial
        )
        self.model.ent_coef = current_ent

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(
                f"Step {self.num_timesteps}/{self.total_timesteps} - Updated ent_coef: {current_ent:.6f}"
            )

        return True


class UnifiedEvalCallback(BaseCallback):
    """
    Evaluates the model, reports metrics to Optuna monotonically across CV folds,
    and saves checkpoints scored by Calmar ratio.
    """

    def __init__(
        self,
        config: RLConfig,
        eval_env: MinimalCryptoEnv | ActionMasker,
        trial: optuna.trial.Trial | None,
        checkpoint_dir: Path,
        fold_idx: int = 0,
        eval_step_offset: int = 0,
    ):
        super().__init__(verbose=0)
        self.config = config
        self.eval_env = eval_env
        self.trial = trial
        self.checkpoint_dir = checkpoint_dir
        self.fold_idx = fold_idx
        self.eval_step_offset = eval_step_offset

        self.eval_freq = config.eval_freq
        self.max_checkpoints = config.max_checkpoints

        self.eval_idx = 0
        self.is_pruned = False
        self.best_calmar = -float("inf")
        self.ema_score = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.eval_idx += 1
            _, calmar = self._run_evaluation()
            # --- NEW: Exponential Moving Average (EMA) Smoothing ---
            # Alpha of 0.5 means the new score is 50% current eval, 50% historical average.
            alpha = 0.5
            if self.ema_score is None:
                self.ema_score = calmar
            else:
                self.ema_score = (alpha * calmar) + ((1.0 - alpha) * self.ema_score)
            if self.trial is not None:
                # Monotonically unique step across folds
                report_step = self.eval_step_offset + self.eval_idx
                # Suppress the harmless Optuna duplicate step warning without forcing a SQLite DB read
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*is already reported.*",
                    )
                    # Report the SMOOTHED score to Optuna's MedianPruner
                    self.trial.report(self.ema_score, report_step)

                if self.trial.should_prune():
                    self.is_pruned = True
                    return False

            if calmar > self.best_calmar:
                self.best_calmar = calmar
                if self.config.checkpoint:
                    self._save_checkpoint(calmar)

        return True

    def _run_evaluation(self) -> tuple[float, float]:
        """Runs one episode to extract both total reward and Calmar ratio."""
        obs = self.eval_env.reset()
        done = False
        episode_reward = 0.0
        # Access the unwrapped base environment for fast property lookups
        # (bypassing slow VecEnv get_attr IPC calls)
        base_env = self.eval_env.venv.envs[0].unwrapped

        initial_pv = base_env.portfolio_value
        portfolio_values = [{"step": 0, "value": float(initial_pv)}]
        steps = 0
        while not done:
            masks = self.eval_env.env_method("action_masks")[0]
            current_masks = np.array([masks])

            action, _ = self.model.predict(
                obs, action_masks=current_masks, deterministic=True
            )

            obs, reward, done_array, infos = self.eval_env.step(action)
            done = done_array[0]
            info = infos[0]
            episode_reward += float(reward[0])
            steps += 1
            # Dodge the VecEnv auto-reset bug
            if done:
                # On the terminal step, grab the true final PV from info before the reset wiped it
                current_pv = info.get("final_portfolio_value", base_env.portfolio_value)
            else:
                # Otherwise, grab the live PV safely
                current_pv = base_env.portfolio_value
            portfolio_values.append(
                {
                    "step": steps,
                    "value": float(current_pv),
                }
            )
        calmar = calculate_calmar_ratio(portfolio_values)
        return episode_reward, calmar

    def _save_checkpoint(self, calmar: float) -> None:
        """Saves the model with fold metadata and trims old checkpoints."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        ckpt_path = (
            self.checkpoint_dir
            / f"fold_{self.fold_idx}_step_{self.num_timesteps}_calmar_{calmar:.4f}_{timestamp}.zip"
        )
        self.model.save(str(ckpt_path))

        all_ckpts = sorted(
            self.checkpoint_dir.parent.parent.glob("**/*.zip"),
            key=lambda p: p.stat().st_mtime,
        )

        if len(all_ckpts) > self.max_checkpoints:
            for old_ckpt in all_ckpts[: -self.max_checkpoints]:
                try:
                    old_ckpt.unlink()
                except Exception as e:
                    print(f"Failed to delete old checkpoint: {e}")
