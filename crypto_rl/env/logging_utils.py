import time
from pathlib import Path
import pandas as pd


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
    if not env.disable_logging and env.log_buffer and env.log_file_path:
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
    if env.disable_logging:
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
        "reward_components": reward_components
    }
    env.log_buffer.append(entry)
    env.last_invalid_sell = False
    env.last_remap_note = None
