#!/usr/bin/env python3
"""Conservative rolling walk-forward experiment runner."""

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

BASE_ARGS = [
    sys.executable,
    "rl_trader.py",
    "--mode",
    "backtest_walk_forward",
    "--device",
    "cpu",
    "--fee-regime",
    "mexc_spot_mx",
    "--action-mapping-mode",
    "legacy",
    "--invalid-sell-mode",
    "force_buy",
    "--trade-action-bonus",
    "8.5",
    "--inactivity-penalty",
    "-0.9",
    "--trade-execution-penalty",
    "0.11",
    "--trade-rate-penalty",
    "0.22",
    "--target-trade-rate",
    "0.11",
    "--trade-rate-window",
    "240",
    "--reward-equity-delta-scale",
    "100.0",
    "--turnover-penalty-rate",
    "0.14",
    "--continuous-drawdown-penalty",
    "5.0",
    "--min-hold-steps",
    "4",
    "--trade-cooldown-steps",
    "1",
    "--max-trades-per-window",
    "96",
    "--trade-window-steps",
    "240",
    "--constraint-violation-penalty",
    "0.20",
    "--randomize-episode-start",
    "--min-episode-steps",
    "1800",
    "--max-episode-steps",
    "5400",
    "--fee-randomization-pct",
    "0.10",
    "--walk-forward-folds",
    "3",
    "--walk-forward-train-ratio",
    "0.50",
    "--walk-forward-val-ratio",
    "0.15",
    "--walk-forward-test-ratio",
    "0.15",
    "--walk-forward-step-ratio",
    "0.10",
    "--model-arch",
    "[384, 192, 96]",
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



def trade_band_ok(metrics: Dict[str, float], lo: int = 100, hi: int = 12000) -> bool:
    return lo <= metrics.get("val_trades", 0.0) <= hi and lo <= metrics.get("test_trades", 0.0) <= hi



def score(metrics: Dict[str, float]) -> float:
    val_ret = metrics.get("val_return_pct", 0.0)
    test_ret = metrics.get("test_return_pct", 0.0)
    gap = abs(val_ret - test_ret)
    return min(val_ret, test_ret) + 0.4 * test_ret - 0.05 * gap



def run_case(case: ExpCase, variant: ConfigVariant, log_dir: Path) -> Dict:
    cmd = with_overrides(BASE_ARGS, variant.overrides)
    cmd.extend([
        "--days",
        str(case.days),
        "--max-symbols",
        str(case.max_symbols),
        "--train-steps",
        str(case.train_steps),
        "--seed",
        str(case.seed),
    ])

    print(f"\n[EXP10] {variant.name} | seed={case.seed} | {case.days}d/{case.max_symbols}s")
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    output = proc.stdout + "\n" + proc.stderr

    metrics = parse_metrics(output)
    result = {
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
    log_path = log_dir / f"exp10_{variant.name}_seed{case.seed}.log"
    log_path.write_text(output)
    result["log_path"] = str(log_path)

    print(
        "[EXP10] done "
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



def parse_seed_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]



def build_variants() -> List[ConfigVariant]:
    return [
        ConfigVariant(
            name="wf_rebalance_midflow",
            overrides={
                "--max-budget-per-trade": "6.5",
                "--model-arch": "[384, 192, 96]",
                "--trade-action-bonus": "9.5",
                "--inactivity-penalty": "-1.1",
                "--trade-execution-penalty": "0.09",
                "--trade-rate-penalty": "0.18",
                "--target-trade-rate": "0.13",
                "--turnover-penalty-rate": "0.12",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "140",
                "--constraint-violation-penalty": "0.14",
                "--reward-equity-delta-scale": "95.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_plus",
            overrides={
                "--max-budget-per-trade": "6.2",
                "--model-arch": "[384, 192, 96]",
                "--trade-action-bonus": "10.0",
                "--inactivity-penalty": "-1.2",
                "--trade-execution-penalty": "0.08",
                "--trade-rate-penalty": "0.16",
                "--target-trade-rate": "0.14",
                "--turnover-penalty-rate": "0.10",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "160",
                "--constraint-violation-penalty": "0.12",
                "--reward-equity-delta-scale": "92.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality",
            overrides={
                "--max-budget-per-trade": "5.8",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.8",
                "--inactivity-penalty": "-0.95",
                "--trade-execution-penalty": "0.11",
                "--trade-rate-penalty": "0.22",
                "--target-trade-rate": "0.11",
                "--turnover-penalty-rate": "0.13",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "120",
                "--constraint-violation-penalty": "0.18",
                "--reward-equity-delta-scale": "105.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2",
            overrides={
                "--max-budget-per-trade": "5.2",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.5",
                "--inactivity-penalty": "-0.9",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.26",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "104",
                "--constraint-violation-penalty": "0.22",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v3",
            overrides={
                "--max-budget-per-trade": "5.8",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "9.2",
                "--inactivity-penalty": "-1.1",
                "--trade-execution-penalty": "0.11",
                "--trade-rate-penalty": "0.20",
                "--target-trade-rate": "0.12",
                "--turnover-penalty-rate": "0.14",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "140",
                "--constraint-violation-penalty": "0.16",
                "--reward-equity-delta-scale": "106.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2_lowbudget",
            overrides={
                "--max-budget-per-trade": "3.5",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.5",
                "--inactivity-penalty": "-0.9",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.26",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "104",
                "--constraint-violation-penalty": "0.22",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2_lowbudget_relax",
            overrides={
                "--max-budget-per-trade": "3.5",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.8",
                "--inactivity-penalty": "-1.0",
                "--trade-execution-penalty": "0.10",
                "--trade-rate-penalty": "0.20",
                "--target-trade-rate": "0.11",
                "--turnover-penalty-rate": "0.12",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "128",
                "--constraint-violation-penalty": "0.18",
                "--reward-equity-delta-scale": "106.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2_relax_tradecap",
            overrides={
                "--max-budget-per-trade": "5.2",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.5",
                "--inactivity-penalty": "-0.9",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.26",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "136",
                "--constraint-violation-penalty": "0.22",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2_relax_ratepen",
            overrides={
                "--max-budget-per-trade": "5.2",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.5",
                "--inactivity-penalty": "-0.9",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.18",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "104",
                "--constraint-violation-penalty": "0.22",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_midflow_quality_v2_relax_turnover",
            overrides={
                "--max-budget-per-trade": "5.2",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.5",
                "--inactivity-penalty": "-0.9",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.26",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.10",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "104",
                "--constraint-violation-penalty": "0.22",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_struct_hold_validity",
            overrides={
                "--max-budget-per-trade": "6.0",
                "--model-arch": "[384, 192, 96]",
                "--action-mapping-mode": "validity_constrained",
                "--invalid-sell-mode": "hold",
                "--trade-action-bonus": "9.2",
                "--inactivity-penalty": "-1.0",
                "--trade-execution-penalty": "0.10",
                "--trade-rate-penalty": "0.18",
                "--target-trade-rate": "0.12",
                "--turnover-penalty-rate": "0.12",
                "--min-hold-steps": "3",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "120",
                "--constraint-violation-penalty": "0.18",
                "--reward-equity-delta-scale": "98.0",
            },
        ),
        ConfigVariant(
            name="wf_struct_hold_active",
            overrides={
                "--max-budget-per-trade": "6.2",
                "--model-arch": "[384, 192, 96]",
                "--action-mapping-mode": "validity_constrained",
                "--invalid-sell-mode": "hold",
                "--trade-action-bonus": "11.5",
                "--inactivity-penalty": "-2.0",
                "--trade-execution-penalty": "0.07",
                "--trade-rate-penalty": "0.14",
                "--target-trade-rate": "0.14",
                "--turnover-penalty-rate": "0.09",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "180",
                "--constraint-violation-penalty": "0.10",
                "--reward-equity-delta-scale": "90.0",
            },
        ),
        ConfigVariant(
            name="wf_inventory_delta_bootstrap",
            overrides={
                "--max-budget-per-trade": "6.0",
                "--model-arch": "[256, 128, 64]",
                "--action-mapping-mode": "inventory_delta",
                "--action-deadzone": "0.04",
                "--sell-redirect-mode": "oldest",
                "--semantic-bootstrap-buy-pct": "0.18",
                "--semantic-bootstrap-penalty": "0.05",
                "--trade-action-bonus": "2.5",
                "--inactivity-penalty": "-0.25",
                "--trade-execution-penalty": "0.10",
                "--trade-rate-penalty": "0.30",
                "--target-trade-rate": "0.10",
                "--min-hold-steps": "2",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "96",
                "--constraint-violation-penalty": "0.15",
                "--reward-equity-delta-scale": "80.0",
                "--turnover-penalty-rate": "0.06",
                "--continuous-drawdown-penalty": "2.5",
            },
        ),
        ConfigVariant(
            name="wf_rebalance_base",
            overrides={
                "--max-budget-per-trade": "6.0",
                "--model-arch": "[384, 192, 96]",
            },
        ),
        ConfigVariant(
            name="wf_rebalance_tighter",
            overrides={
                "--max-budget-per-trade": "5.0",
                "--model-arch": "[384, 192, 96]",
                "--trade-action-bonus": "7.2",
                "--inactivity-penalty": "-0.7",
                "--trade-execution-penalty": "0.14",
                "--trade-rate-penalty": "0.30",
                "--target-trade-rate": "0.09",
                "--turnover-penalty-rate": "0.18",
                "--min-hold-steps": "5",
                "--trade-cooldown-steps": "2",
                "--max-trades-per-window": "80",
                "--constraint-violation-penalty": "0.30",
                "--reward-equity-delta-scale": "112.0",
            },
        ),
        ConfigVariant(
            name="wf_rebalance_capacity",
            overrides={
                "--max-budget-per-trade": "5.0",
                "--model-arch": "[512, 256, 128]",
                "--trade-action-bonus": "8.0",
                "--inactivity-penalty": "-0.8",
                "--trade-execution-penalty": "0.13",
                "--trade-rate-penalty": "0.24",
                "--target-trade-rate": "0.10",
                "--turnover-penalty-rate": "0.16",
                "--min-hold-steps": "4",
                "--trade-cooldown-steps": "1",
                "--max-trades-per-window": "88",
                "--constraint-violation-penalty": "0.24",
                "--reward-equity-delta-scale": "108.0",
            },
        ),
    ]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conservative rolling walk-forward experiments.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--seeds", default="11,23")
    parser.add_argument("--max-variants", type=int, default=4)
    parser.add_argument("--variants", default="", help="Comma-separated variant names to run (default: use first max-variants)")
    parser.add_argument("--log-dir", default="artifacts/experiments/logs")
    parser.add_argument("--results-dir", default="artifacts/experiments/results")
    parser.add_argument("--out", default="exp10_walk_forward_results.json")
    return parser.parse_args()



def main() -> None:
    if ".conda" not in Path(sys.executable).as_posix():
        print(f"[EXP10] Warning: expected project-local interpreter under .conda, got {sys.executable}")

    args = parse_args()
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_variants = build_variants()
    selected_names = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    if selected_names:
        name_set = set(selected_names)
        variants = [v for v in all_variants if v.name in name_set]
        if not variants:
            raise ValueError(f"No variants matched --variants={selected_names}")
    else:
        variants = all_variants[: args.max_variants]
    all_results: List[Dict] = []

    out_path = Path(args.out)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = results_dir / out_path

    for variant in variants:
        for seed in parse_seed_csv(args.seeds):
            case = ExpCase(
                config=variant.name,
                days=args.days,
                max_symbols=args.max_symbols,
                train_steps=args.train_steps,
                seed=seed,
            )
            all_results.append(run_case(case, variant, log_dir))

    out = {
        "created_at_epoch": time.time(),
        "aggregate": aggregate_by_config(all_results),
        "results": all_results,
    }
    out_path.write_text(json.dumps(out, indent=2))

    print("\n[EXP10] Results written to", out_path)
    print("[EXP10] Ranking:")
    for row in out["aggregate"]:
        print(
            "  "
            f"{row['config']}: med_val={row['median_val_return_pct']:.3f}% "
            f"med_test={row['median_test_return_pct']:.3f}% "
            f"med_min={row['median_min_return_pct']:.3f}% trade_ok={row['trade_band_ok_rate']:.2f}"
        )


if __name__ == "__main__":
    main()

