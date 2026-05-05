#!/usr/bin/env python3
"""Two-stage recovery curriculum runner with trade-min promotion gating."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

VAL_RE = re.compile(r"Val Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")
TEST_RE = re.compile(r"Test Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")


@dataclass
class StageResult:
    name: str
    command: List[str]
    log_file: str
    return_code: int
    elapsed_seconds: float
    metrics: Dict[str, float]


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_metrics(output: str) -> Dict[str, float]:
    metrics = {
        "val_trades": 0.0,
        "val_win_rate_pct": 0.0,
        "val_return_pct": 0.0,
        "test_trades": 0.0,
        "test_win_rate_pct": 0.0,
        "test_return_pct": 0.0,
    }
    val_matches = VAL_RE.findall(output)
    if val_matches:
        trades, wr, ret = val_matches[-1]
        metrics["val_trades"] = float(trades)
        metrics["val_win_rate_pct"] = float(wr)
        metrics["val_return_pct"] = float(ret)
    test_matches = TEST_RE.findall(output)
    if test_matches:
        trades, wr, ret = test_matches[-1]
        metrics["test_trades"] = float(trades)
        metrics["test_win_rate_pct"] = float(wr)
        metrics["test_return_pct"] = float(ret)
    return metrics


def _run_stage(stage_name: str, cmd: List[str], log_path: Path) -> StageResult:
    print(f"[CURRICULUM] {stage_name} command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    metrics = _parse_metrics(output)
    print(
        f"[CURRICULUM] {stage_name} done rc={proc.returncode} "
        f"val_trades={int(metrics['val_trades'])} test_trades={int(metrics['test_trades'])} "
        f"val_ret={metrics['val_return_pct']:.2f}% test_ret={metrics['test_return_pct']:.2f}% "
        f"elapsed={elapsed:.1f}s"
    )
    return StageResult(
        name=stage_name,
        command=cmd,
        log_file=str(log_path),
        return_code=int(proc.returncode),
        elapsed_seconds=float(elapsed),
        metrics=metrics,
    )


def _base_args(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "rl_trader.py",
        "--mode", "backtest_3way",
        "--algo", "ppo",
        "--device", args.device,
        "--days", str(args.days),
        "--max-symbols", str(args.max_symbols),
        "--train-steps", str(args.train_steps),
        "--eval-freq", str(args.eval_freq),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--n-steps", str(args.n_steps),
        "--n-epochs", str(args.n_epochs),
        "--fee-regime", "mexc_spot_mx",
        "--action-mapping-mode", "inventory_delta",
        "--action-deadzone", "0.04",
        "--sell-redirect-mode", "oldest",
        "--semantic-bootstrap-buy-pct", "0.18",
        "--semantic-bootstrap-penalty", "0.0",
        "--trade-rate-window", "240",
        "--target-trade-rate", "0.10",
        "--trade-rate-penalty", "0.15",
        "--min-hold-steps", "2",
        "--trade-cooldown-steps", "1",
        "--max-trades-per-window", "96",
        "--trade-window-steps", "240",
        "--constraint-violation-penalty", "0.15",
        "--reward-equity-delta-scale", "80.0",
        "--turnover-penalty-rate", "0.04",
        "--continuous-drawdown-penalty", "2.5",
        "--randomize-episode-start",
        "--min-episode-steps", "1800",
        "--max-episode-steps", "5400",
        "--fee-randomization-pct", "0.10",
        "--model-arch", "[256, 128, 64]",
        "--dashboard",
        "--dashboard-port", str(args.dashboard_port),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage trade-activation curriculum.")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--max-symbols", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=180000)
    parser.add_argument("--eval-freq", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--dashboard-port", type=int, default=8772)
    parser.add_argument("--min-val-trades", type=int, default=6)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--notes", type=str, default="curriculum_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = _utc_ts()
    run_name = f"{stamp}_recovery_curriculum_v1"
    artifact_root = Path("artifacts") / "experiments"
    launch_path = artifact_root / "launches" / f"{run_name}.json"
    summary_path = artifact_root / "summaries" / f"{run_name}.json"
    stage1_log = artifact_root / "logs" / f"{run_name}_stage1_shaped.log"
    stage2_log = artifact_root / "logs" / f"{run_name}_stage2_equity.log"

    base = _base_args(args)
    stage1_state = f"rl_dashboard_state_{run_name}_stage1.json"
    stage2_state = f"rl_dashboard_state_{run_name}_stage2.json"

    stage1_cmd = base + [
        "--reward-mode", "shaped",
        "--ent-coef", "0.015",
        "--inactivity-penalty", "-3.0",
        "--dashboard-state", stage1_state,
    ]
    stage2_cmd = base + [
        "--reward-mode", "equity_delta",
        "--ent-coef", "0.005",
        "--inactivity-penalty", "-1.2",
        "--init-model", "models/final_model",
        "--dashboard-state", stage2_state,
    ]

    _write_json(
        launch_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": "run_recovery_curriculum.py",
            "notes": args.notes,
            "trade_gate": {
                "min_val_trades": int(args.min_val_trades),
                "min_test_trades": int(args.min_test_trades),
            },
            "stage1_command": stage1_cmd,
            "stage2_command": stage2_cmd,
        },
    )
    print(f"[CURRICULUM] Launch manifest: {launch_path}")

    stage1 = _run_stage("stage1_shaped_activation", stage1_cmd, stage1_log)
    if stage1.return_code != 0:
        _write_json(
            summary_path,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner": "run_recovery_curriculum.py",
                "result": "failed_stage1",
                "stage1": stage1.__dict__,
            },
        )
        sys.exit(stage1.return_code)

    gate_pass = (
        int(stage1.metrics["val_trades"]) >= int(args.min_val_trades)
        and int(stage1.metrics["test_trades"]) >= int(args.min_test_trades)
    )
    if not gate_pass:
        _write_json(
            summary_path,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner": "run_recovery_curriculum.py",
                "result": "blocked_by_trade_gate",
                "trade_gate": {
                    "min_val_trades": int(args.min_val_trades),
                    "min_test_trades": int(args.min_test_trades),
                    "pass": False,
                },
                "stage1": stage1.__dict__,
            },
        )
        print("[CURRICULUM] Stage 1 failed trade gate; skipping Stage 2.")
        sys.exit(2)

    print("[CURRICULUM] Stage 1 passed trade gate; running Stage 2 transfer.")
    stage2 = _run_stage("stage2_equity_transfer", stage2_cmd, stage2_log)

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "run_recovery_curriculum.py",
        "result": "completed" if stage2.return_code == 0 else "failed_stage2",
        "trade_gate": {
            "min_val_trades": int(args.min_val_trades),
            "min_test_trades": int(args.min_test_trades),
            "pass": True,
        },
        "stage1": stage1.__dict__,
        "stage2": stage2.__dict__,
    }
    _write_json(summary_path, summary)
    print(f"[CURRICULUM] Summary: {summary_path}")
    if stage2.return_code != 0:
        sys.exit(stage2.return_code)


if __name__ == "__main__":
    main()

