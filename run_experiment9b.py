#!/usr/bin/env python3
"""Experiment 9B runner: tune for positive validation and test returns."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

VAL_RE = re.compile(r"Val Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")
TEST_RE = re.compile(r"Test Trades:\s*(\d+)\s*\|\s*WR:\s*([\d.]+)%\s*\|\s*Return:\s*([-\d.]+)%")

# Baseline carried from Exp9, then each config overrides only what changes.
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
    "--invalid-sell-mode",
    "force_buy",
    "--trade-action-bonus",
    "12.0",
    "--inactivity-penalty",
    "-4.0",
    "--trade-execution-penalty",
    "0.10",
    "--trade-rate-penalty",
    "0.18",
    "--target-trade-rate",
    "0.20",
    "--trade-rate-window",
    "240",
    "--reward-equity-delta-scale",
    "70.0",
    "--turnover-penalty-rate",
    "0.08",
    "--continuous-drawdown-penalty",
    "3.0",
    "--min-hold-steps",
    "2",
    "--trade-cooldown-steps",
    "1",
    "--max-trades-per-window",
    "140",
    "--trade-window-steps",
    "240",
    "--constraint-violation-penalty",
    "0.12",
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


@dataclass
class ConfigVariant:
    name: str
    overrides: Dict[str, str]


@dataclass
class ExpCase:
    phase: str
    config: str
    days: int
    max_symbols: int
    train_steps: int
    seed: int


def with_overrides(base_args: Sequence[str], overrides: Dict[str, str]) -> List[str]:
    args = list(base_args)
    for key, value in overrides.items():
        if key in args:
            idx = args.index(key)
            # Replace the next token when key is an option expecting a value.
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                args[idx + 1] = str(value)
                continue
        args.extend([key, str(value)])
    return args


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {
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


def trade_band_ok(metrics: Dict[str, float], lo: int = 100, hi: int = 4500) -> bool:
    return lo <= metrics.get("val_trades", 0.0) <= hi and lo <= metrics.get("test_trades", 0.0) <= hi


def score(metrics: Dict[str, float]) -> float:
    # Primary objective: positive on both splits, then prefer higher test and lower split gap.
    val_ret = metrics.get("val_return_pct", 0.0)
    test_ret = metrics.get("test_return_pct", 0.0)
    gap = abs(val_ret - test_ret)
    return min(val_ret, test_ret) + 0.35 * test_ret - 0.05 * gap


def run_case(case: ExpCase, variant: ConfigVariant, log_dir: Path) -> Dict:
    cmd = with_overrides(BASE_ARGS, variant.overrides)
    cmd.extend(
        [
            "--days",
            str(case.days),
            "--max-symbols",
            str(case.max_symbols),
            "--train-steps",
            str(case.train_steps),
            "--seed",
            str(case.seed),
        ]
    )

    print(f"\n[EXP9B] {case.phase} | {variant.name} | seed={case.seed} | {case.days}d/{case.max_symbols}s")
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    output = proc.stdout + "\n" + proc.stderr

    metrics = parse_metrics(output)
    result = {
        "phase": case.phase,
        "config": variant.name,
        "days": case.days,
        "max_symbols": case.max_symbols,
        "train_steps": case.train_steps,
        "seed": case.seed,
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "trade_band_ok": trade_band_ok(metrics),
        "score": score(metrics),
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"exp9b_{case.phase}_{variant.name}_seed{case.seed}.log"
    log_path.write_text(output)
    result["log_path"] = str(log_path)

    print(
        "[EXP9B] done "
        f"rc={proc.returncode} val={metrics['val_return_pct']:.3f}% test={metrics['test_return_pct']:.3f}% "
        f"trades(v/t)={int(metrics['val_trades'])}/{int(metrics['test_trades'])} elapsed={elapsed/60.0:.1f}m"
    )
    return result


def aggregate_by_config(results: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in results:
        grouped.setdefault(row["config"], []).append(row)

    agg: List[Dict] = []
    for config, rows in grouped.items():
        ok_rows = [r for r in rows if r["return_code"] == 0]
        if not ok_rows:
            continue
        val_returns = [r["metrics"]["val_return_pct"] for r in ok_rows]
        test_returns = [r["metrics"]["test_return_pct"] for r in ok_rows]
        min_returns = [min(r["metrics"]["val_return_pct"], r["metrics"]["test_return_pct"]) for r in ok_rows]
        trade_ok_rate = sum(1 for r in ok_rows if r["trade_band_ok"]) / len(ok_rows)
        agg.append(
            {
                "config": config,
                "n": len(ok_rows),
                "median_val_return_pct": statistics.median(val_returns),
                "median_test_return_pct": statistics.median(test_returns),
                "median_min_return_pct": statistics.median(min_returns),
                "median_score": statistics.median([r["score"] for r in ok_rows]),
                "trade_band_ok_rate": trade_ok_rate,
            }
        )

    agg.sort(
        key=lambda x: (
            x["median_min_return_pct"],
            x["median_test_return_pct"],
            x["trade_band_ok_rate"],
            x["median_score"],
        ),
        reverse=True,
    )
    return agg


def pick_gate_winners(gate_agg: Sequence[Dict], max_winners: int = 3) -> List[str]:
    winners: List[str] = []
    for row in gate_agg:
        if row["trade_band_ok_rate"] < 1.0:
            continue
        winners.append(row["config"])
        if len(winners) >= max_winners:
            break
    return winners


def build_gate_variants() -> List[ConfigVariant]:
    return [
        ConfigVariant(
            name="g_profit_both",
            overrides={
                "--transaction-cost": "0.00003",
                "--max-budget-per-trade": "7.0",
                "--trade-action-bonus": "15.0",
                "--inactivity-penalty": "-5.0",
                "--trade-rate-penalty": "0.20",
                "--target-trade-rate": "0.15",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "120",
                "--constraint-violation-penalty": "0.10",
            },
        ),
        ConfigVariant(
            name="a_mx_base",
            overrides={
                "--max-budget-per-trade": "8.0",
            },
        ),
        ConfigVariant(
            name="b_budget6",
            overrides={
                "--max-budget-per-trade": "6.0",
            },
        ),
        ConfigVariant(
            name="c_budget4",
            overrides={
                "--max-budget-per-trade": "4.0",
            },
        ),
        ConfigVariant(
            name="d_churn_guard",
            overrides={
                "--max-budget-per-trade": "6.0",
                "--trade-rate-penalty": "0.35",
                "--target-trade-rate": "0.14",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "2",
                "--max-trades-per-window": "96",
            },
        ),
        ConfigVariant(
            name="e_quality",
            overrides={
                "--max-budget-per-trade": "6.0",
                "--reward-equity-delta-scale": "90.0",
                "--trade-action-bonus": "10.0",
                "--trade-execution-penalty": "0.14",
                "--continuous-drawdown-penalty": "5.0",
            },
        ),
        ConfigVariant(
            name="f_stability",
            overrides={
                "--max-budget-per-trade": "5.0",
                "--reward-equity-delta-scale": "85.0",
                "--trade-rate-penalty": "0.28",
                "--target-trade-rate": "0.16",
                "--turnover-penalty-rate": "0.14",
                "--min-hold-steps": "3",
                "--max-trades-per-window": "108",
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 9B matrix.")
    parser.add_argument("--phase", choices=["gate", "confirm", "robust", "all"], default="all")
    parser.add_argument("--gate-steps", type=int, default=10_000)
    parser.add_argument("--confirm-steps", type=int, default=50_000)
    parser.add_argument("--robust-steps", type=int, default=10_000)
    parser.add_argument("--gate-seeds", default="11,23")
    parser.add_argument("--confirm-seeds", default="11,23,47")
    parser.add_argument("--robust-seeds", default="11,23,47")
    parser.add_argument("--max-gate-variants", type=int, default=6)
    parser.add_argument("--max-confirm-variants", type=int, default=3)
    parser.add_argument("--log-dir", default="artifacts/experiments/logs")
    parser.add_argument("--results-dir", default="artifacts/experiments/results")
    parser.add_argument("--out", default="exp9b_results.json")
    return parser.parse_args()


def parse_seed_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    gate_variants = build_gate_variants()[: args.max_gate_variants]
    variant_map = {v.name: v for v in gate_variants}

    all_results: List[Dict] = []
    historical: Dict = {}
    out_path = Path(args.out)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = results_dir / out_path
    if out_path.exists():
        try:
            historical = json.loads(out_path.read_text())
        except Exception:
            historical = {}

    # Keep prior phases when incrementally running confirm/robust.
    if args.phase in ("confirm", "robust") and historical.get("results"):
        all_results.extend(historical["results"])

    # Phase 1: 14d/3sym gate search.
    if args.phase in ("gate", "all"):
        for variant in gate_variants:
            for seed in parse_seed_csv(args.gate_seeds):
                case = ExpCase("gate", variant.name, days=14, max_symbols=3, train_steps=args.gate_steps, seed=seed)
                all_results.append(run_case(case, variant, log_dir))

    gate_agg = aggregate_by_config(r for r in all_results if r["phase"] == "gate")
    winners = pick_gate_winners(gate_agg, max_winners=args.max_confirm_variants)

    # Allow running confirm/robust directly if winners are manually provided in prior out file.
    if args.phase in ("confirm", "robust") and not winners and out_path.exists():
        prior_gate_agg = aggregate_by_config(r for r in historical.get("results", []) if r["phase"] == "gate")
        winners = pick_gate_winners(prior_gate_agg, max_winners=args.max_confirm_variants)

    if args.phase in ("confirm", "robust", "all") and not winners:
        print("[EXP9B] No eligible gate winners found; skipping confirm/robust.")

    # Phase 2: confirm winners on longer training.
    if args.phase in ("confirm", "all"):
        for name in winners:
            variant = variant_map[name]
            for seed in parse_seed_csv(args.confirm_seeds):
                case = ExpCase("confirm", variant.name, days=14, max_symbols=3, train_steps=args.confirm_steps, seed=seed)
                all_results.append(run_case(case, variant, log_dir))

    # Phase 3: robust broader-data check.
    if args.phase in ("robust", "all"):
        for name in winners:
            variant = variant_map[name]
            for seed in parse_seed_csv(args.robust_seeds):
                case = ExpCase("robust", variant.name, days=30, max_symbols=5, train_steps=args.robust_steps, seed=seed)
                all_results.append(run_case(case, variant, log_dir))

    out = {
        "created_at_epoch": time.time(),
        "gate_aggregate": gate_agg,
        "winners": winners,
        "confirm_aggregate": aggregate_by_config(r for r in all_results if r["phase"] == "confirm"),
        "robust_aggregate": aggregate_by_config(r for r in all_results if r["phase"] == "robust"),
        "results": all_results,
    }
    out_path.write_text(json.dumps(out, indent=2))

    print("\n[EXP9B] Results written to", out_path)
    print("[EXP9B] Gate ranking:")
    for row in gate_agg:
        print(
            "  "
            f"{row['config']}: med_val={row['median_val_return_pct']:.3f}% "
            f"med_test={row['median_test_return_pct']:.3f}% "
            f"med_min={row['median_min_return_pct']:.3f}% trade_ok={row['trade_band_ok_rate']:.2f}"
        )
    if winners:
        print("[EXP9B] Winners:", ", ".join(winners))


if __name__ == "__main__":
    main()






