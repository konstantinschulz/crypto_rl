#!/usr/bin/env python3
"""Experiment 9 runner: profitability + robustness sweeps across seeds and broader data."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


RETURN_RE = re.compile(r"Test Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")


@dataclass
class Exp9Case:
    name: str
    days: int
    max_symbols: int
    train_steps: int
    seed: int
    extra: List[str]


BASE_ARGS = [
    sys.executable,
    "rl_trader.py",
    "--mode",
    "backtest_3way",
    "--device",
    "cpu",
    "--fee-regime",
    "mexc_spot_mx",
    "--action-mapping-mode",
    "legacy",
    "--trade-action-bonus",
    "1.2",
    "--inactivity-penalty",
    "-0.35",
    "--trade-execution-penalty",
    "0.08",
    "--trade-rate-penalty",
    "0.35",
    "--target-trade-rate",
    "0.14",
    "--trade-rate-window",
    "240",
    "--reward-equity-delta-scale",
    "120.0",
    "--turnover-penalty-rate",
    "0.14",
    "--continuous-drawdown-penalty",
    "6.0",
    "--min-hold-steps",
    "3",
    "--trade-cooldown-steps",
    "1",
    "--max-trades-per-window",
    "96",
    "--trade-window-steps",
    "240",
    "--constraint-violation-penalty",
    "0.25",
    "--randomize-episode-start",
    "--min-episode-steps",
    "1200",
    "--max-episode-steps",
    "3600",
    "--fee-randomization-pct",
    "0.20",
    "--model-arch",
    "[256, 128, 64]",
    "--learning-rate",
    "0.0002",
    "--batch-size",
    "16",
    "--n-steps",
    "512",
    "--n-epochs",
    "5",
]


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "test_trades": 0.0,
        "test_win_rate_pct": 0.0,
        "test_return_pct": 0.0,
    }
    matches = RETURN_RE.findall(output)
    if not matches:
        return metrics

    trades, wr, ret = matches[-1]
    metrics["test_trades"] = float(trades)
    metrics["test_win_rate_pct"] = float(wr)
    metrics["test_return_pct"] = float(ret)
    return metrics


def run_case(case: Exp9Case, log_dir: Path) -> Dict:
    cmd = BASE_ARGS + [
        "--days",
        str(case.days),
        "--max-symbols",
        str(case.max_symbols),
        "--train-steps",
        str(case.train_steps),
        "--seed",
        str(case.seed),
    ] + case.extra

    print(f"\n[EXP9] Running {case.name}")
    print("[EXP9] Command:", " ".join(cmd))

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    output = proc.stdout + "\n" + proc.stderr

    metrics = parse_metrics(output)
    result = {
        "name": case.name,
        "days": case.days,
        "max_symbols": case.max_symbols,
        "train_steps": case.train_steps,
        "seed": case.seed,
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "metrics": metrics,
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"exp9_{case.name.replace(' ', '_').lower()}.log"
    log_path.write_text(output)
    result["log_path"] = str(log_path)
    print(
        f"[EXP9] done rc={proc.returncode} return={metrics['test_return_pct']:.3f}% "
        f"trades={int(metrics['test_trades'])} elapsed={elapsed/60.0:.1f}m"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 9 sweep.")
    parser.add_argument("--log-dir", default="artifacts/experiments/logs")
    parser.add_argument("--results-dir", default="artifacts/experiments/results")
    parser.add_argument("--out", default="exp9_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = [11, 23, 47]
    cases: List[Exp9Case] = []
    for seed in seeds:
        cases.append(
            Exp9Case(
                name=f"core_14d3s_seed{seed}",
                days=14,
                max_symbols=3,
                train_steps=10_000,
                seed=seed,
                extra=[],
            )
        )
        cases.append(
            Exp9Case(
                name=f"robust_30d5s_seed{seed}",
                days=30,
                max_symbols=5,
                train_steps=10_000,
                seed=seed,
                extra=[],
            )
        )

    all_results = [run_case(case, log_dir) for case in cases]
    out_path = Path(args.out)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = results_dir / out_path
    out_path.write_text(json.dumps(all_results, indent=2))

    returns = [r["metrics"].get("test_return_pct", 0.0) for r in all_results if r["return_code"] == 0]
    robust_returns = [
        r["metrics"].get("test_return_pct", 0.0)
        for r in all_results
        if r["return_code"] == 0 and "robust_30d5s" in r["name"]
    ]

    print("\n[EXP9] Results written to", out_path)
    if returns:
        print(f"[EXP9] median return (all successful): {sorted(returns)[len(returns)//2]:.3f}%")
    if robust_returns:
        print(f"[EXP9] median robust return (30d/5sym): {sorted(robust_returns)[len(robust_returns)//2]:.3f}%")


if __name__ == "__main__":
    main()

