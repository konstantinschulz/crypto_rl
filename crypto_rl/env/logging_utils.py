import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import gi
import optuna
import pandas as pd

gi.require_version("Notify", "0.7")
from gi.repository import GLib, Notify

Notify.init("crypto_rl")  # refers to ~/.local/share/applications/crypto_rl.desktop


class LoggerBase:
    def debug(self, message: str):
        raise NotImplementedError("Subclasses must implement the 'debug' method.")

    def info(self, message: str):
        raise NotImplementedError("Subclasses must implement the 'info' method.")

    def log(self, message: str, trial: optuna.Trial | None = None):
        raise NotImplementedError("Subclasses must implement the 'log' method.")

    def print_if_not_trial(
        self,
        log_level: int,
        trial: optuna.trial.Trial | None = None,
        msg: str = "",
    ):
        raise NotImplementedError(
            "Subclasses must implement the 'print_if_not_trial' method."
        )


def init_log(env, run_id: str = "default") -> None:
    """Initialize per-run action log path and clear in-memory log buffer."""
    log_root = Path("logs")
    run_log_dir = log_root / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    prefix = "actions_eval" if env.is_eval else "actions"
    env.log_file_path = (
        run_log_dir / f"{prefix}_ep{env.episode_count}_{int(time.time())}.parquet"
    )
    env.log_buffer = []


def flush_log_parquet(env) -> None:
    """Write buffered log entries to a single Parquet file at the end of an episode."""
    if not env.config.disable_logging and env.log_buffer and env.log_file_path:
        try:
            df_log = pd.DataFrame(env.log_buffer)
            df_log.to_parquet(env.log_file_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to write action log to {env.log_file_path}: {e}")
        env.log_buffer = []


def log_action(
    env,
    step: int,
    action,  # np.ndarray
    reward: float,
    portfolio: float,
    trade_price: float = 0.0,
    trade_units: float = 0.0,
    fee: float = 0.0,
    reward_components: dict | None = None,
) -> None:
    """Record step details into the in-memory log buffer."""
    if env.config.disable_logging:
        return
    action_type_idx = int(action[0])
    action_types = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_type_str = action_types.get(action_type_idx, "UNKNOWN")
    asset_idx = int(action[1])
    asset_str = (
        env.asset_names[asset_idx]
        if 0 <= asset_idx < len(env.asset_names)
        else "UNKNOWN"
    )
    entry = {
        "episode": int(env.episode_count),
        "step": int(step),
        "action_type": action_type_str,
        "symbol": asset_str,
        "amount_pct": float(action[2]),
        "reward": float(reward),
        "portfolio": float(portfolio),
        "note": env.last_remap_note if env.last_remap_note else "",
        "price": float(trade_price),
        "units": float(trade_units),
        "fee": float(fee),
        "reward_components": reward_components,
    }
    env.log_buffer.append(entry)
    env.last_invalid_sell = False
    env.last_remap_note = None


@contextmanager
def selective_logger(file_name: str) -> Generator[LoggerBase, None, None]:
    """Context manager providing methods for dual-output or console-only logging."""
    original_stdout = sys.stdout
    os.makedirs(Path(file_name).parent, exist_ok=True)
    log_file = (
        open(file_name, "w", encoding="utf-8")
        if file_name
        else open("/dev/null", "w", encoding="utf-8")
    )

    class Logger(LoggerBase):
        def add_timestamp(self, message: str) -> str:
            """Prepends a UTC timestamp to the log message."""
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            return f"[{timestamp}] {message}"

        def debug(self, message: str):
            """Prints to the console ONLY (hidden from file)."""
            original_stdout.write(self.add_timestamp(message) + "\n")
            original_stdout.flush()

        def flush(self):
            original_stdout.flush()
            log_file.flush()

        def info(self, message: str):
            """Prints to BOTH the console and the file."""
            msg_with_timestamp: str = self.add_timestamp(message)
            original_stdout.write(msg_with_timestamp + "\n")
            log_file.write(msg_with_timestamp + "\n")
            self.flush()

        def log(self, message: str, trial: optuna.Trial | None = None):
            """Writes to the log file, and optionally to the console."""
            msg_with_timestamp: str = self.add_timestamp(message)
            log_file.write(msg_with_timestamp + "\n")
            log_file.flush()
            if trial is None:
                self.debug(msg_with_timestamp)

        def print_if_not_trial(
            self,
            log_level: int,
            trial: optuna.trial.Trial | None = None,
            msg: str = "",
        ):
            if trial is None:
                if log_level == logging.INFO:
                    self.info(msg)
                elif log_level == logging.DEBUG:
                    self.debug(msg)

    logger = Logger()
    try:
        yield logger
    finally:
        log_file.close()


def send_notification(body: str, summary: str = "Experiment Run"):
    notification = Notify.Notification.new(summary, body)
    # Attach the desktop-entry hint so GNOME Shell groups it
    notification.set_hint("desktop-entry", GLib.Variant("s", "crypto_rl"))
    notification.show()
