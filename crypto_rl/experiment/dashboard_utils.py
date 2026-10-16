# -*- coding: utf-8 -*-
"""Dashboard utilities for the experiment.

The original `run_experiment` function performed a fairly large amount of
book‑keeping to update the JSON dashboard state after training. The helper
functions below encapsulate that logic so the main runner stays concise.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_rl.config import RLConfig


def _downsample(series_list: list[dict[str, Any]], max_pts: int = 1000) -> list[dict[str, Any]]:
    """Down‑sample a list of series points to at most *max_pts* entries.

    The function keeps the first *max_pts* equally‑spaced points and ensures the
    final point is retained.
    """
    if not series_list or len(series_list) <= max_pts:
        return series_list
    step_sz = len(series_list) / max_pts
    res = [series_list[int(i * step_sz)] for i in range(max_pts)]
    if res[-1] != series_list[-1]:
        res[-1] = series_list[-1]
    return res


def _json_default(obj: Any) -> Any:
    """Fallback serializer for types the stdlib JSON encoder rejects."""
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, data: Any) -> None:
    """Serialize fully before touching the file, then swap it in atomically.

    Dumping straight into an open ``"w"`` handle truncates the file first, so
    any serialization error leaves a half-written, unreadable ``state.json``.
    """
    payload = json.dumps(data, indent=2, default=_json_default)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp_path, path)


def update_dashboard_state(
    state_file: Path,
    config: RLConfig,
    cv_mean_pv: float,
    cv_mean_pnl: float,
    cv_mean_calmar: float,
    cv_mean_win_rate: float,
    cv_total_trades: int,
    cv_total_fees: float,
    fold_results: list[dict[str, Any]],
    per_asset_stats: dict[str, dict[str, Any]],
    last_eval_reward_totals: dict[str, Any],
    last_eval_steps: int,
    last_eval_portfolio_values: list[dict[str, Any]],
    last_eval_realized_pnl: list[dict[str, Any]],
    buy_hold_baseline: float | None = None,
) -> None:
    """Write the evaluation results back to the dashboard ``state.json``.

    The function mirrors the behaviour that used to live inside
    ``run_experiment``. It safely reads the existing JSON, updates the relevant
    fields and writes the file back with pretty indentation.
    """
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        # If the file does not exist or is corrupted we start with a fresh
        # structure – the dashboard will still be able to render the results.
        state = {}

    # Basic run metadata
    state.setdefault("run", {})["status"] = "evaluated"
    state["run"]["finished_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    baseline = float(buy_hold_baseline) if buy_hold_baseline is not None else float(cv_mean_pv)

    # Finance / evaluation results
    finance = state.setdefault("finance", {})
    finance["evaluation_results"] = {
        "final_portfolio_value": cv_mean_pv,
        "pnl": cv_mean_pnl,
        "evaluation_steps": int(last_eval_steps),
        "eval_trades": int(cv_total_trades),
        "eval_win_rate_pct": float(cv_mean_win_rate),
        "buy_hold_baseline": baseline,
        "total_fees_paid": float(cv_total_fees),
        "cv_folds": fold_results,
        "per_asset_breakdown": per_asset_stats,
    }
    finance["calmar"] = float(cv_mean_calmar)

    # Explainability – cumulative rewards and hyper‑parameters
    explain = state.setdefault("explainability", {})
    explain["cumulative_rewards"] = {k: float(v) for k, v in last_eval_reward_totals.items()}
    explain["hyperparameters"] = config.to_dict()

    # Series data – down‑sampled for size efficiency
    series = state.setdefault("series", {})
    series["test_portfolio_value"] = _downsample(last_eval_portfolio_values)
    series["test_realized_pnl"] = _downsample(last_eval_realized_pnl)

    # Persist the updated JSON
    _write_json_atomic(state_file, state)
