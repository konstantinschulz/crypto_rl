#!/usr/bin/env python3
"""Profile-based launcher for longer RL training runs with dashboard telemetry.

Operational note:
- Enable --dashboard for longer/larger runs so split metrics and recovery diagnostics
  are visible while training/backtesting.
- Skip dashboard for smoke tests and quick local checks to reduce overhead.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _extract_arg(argv: list[str], key: str, default: str = "") -> str:
    if key not in argv:
        return default
    idx = argv.index(key)
    if idx + 1 >= len(argv):
        return default
    value = argv[idx + 1]
    if value.startswith("--"):
        return default
    return value


def _derive_experiment_tags(argv: list[str], profile: str, objective: str) -> dict[str, str]:
    return {
        "profile": profile,
        "objective": objective,
        "mapping_family": _extract_arg(argv, "--action-mapping-mode", "legacy"),
        "reward_mode": _extract_arg(argv, "--reward-mode", "equity_delta"),
        "algo": _extract_arg(argv, "--algo", "ppo"),
        "eval_mode": _extract_arg(argv, "--mode", "train"),
    }


def _profile_args(profile: str) -> list[str]:
    profiles = {
        # Fast pipeline check for dashboard + telemetry plumbing.
        "smoke": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "14",
            "--max-symbols", "5",
            "--train-steps", "4000",
            "--eval-freq", "1000",
            "--learning-rate", "0.0003",
            "--batch-size", "64",
            "--n-steps", "512",
            "--n-epochs", "5",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.12",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--trade-rate-window", "180",
            "--target-trade-rate", "0.16",
            "--trade-rate-penalty", "0.20",
            "--reward-equity-delta-scale", "100.0",
            "--turnover-penalty-rate", "0.05",
            "--continuous-drawdown-penalty", "2.0",
            "--model-arch", "[256, 128, 64]",
        ],
        # Recommended first longer run: big enough to learn, still tractable for a first pass.
        "first_long": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "600000",
            "--eval-freq", "12000",
            "--learning-rate", "0.0002",
            "--batch-size", "256",
            "--n-steps", "2048",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.12",
            "--sell-redirect-mode", "largest_notional",
            "--min-hold-steps", "3",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "72",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.1",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.12",
            "--trade-rate-penalty", "0.40",
            "--reward-equity-delta-scale", "120.0",
            "--turnover-penalty-rate", "0.08",
            "--continuous-drawdown-penalty", "3.0",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "7200",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[256, 128, 64]",
        ],
        # Recovery profile after no-trade collapse: more policy movement + less restrictive mapping.
        "first_long_recovery": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "800000",
            "--eval-freq", "10000",
            "--learning-rate", "0.0003",
            "--batch-size", "128",
            "--n-steps", "1024",
            "--n-epochs", "10",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "legacy",
            "--invalid-sell-mode", "force_buy",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.18",
            "--trade-rate-penalty", "0.12",
            "--reward-equity-delta-scale", "140.0",
            "--turnover-penalty-rate", "0.03",
            "--continuous-drawdown-penalty", "2.0",
            "--randomize-episode-start",
            "--min-episode-steps", "2200",
            "--max-episode-steps", "7200",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[384, 192, 96]",
        ],
        # Recovery profile A: split-eval PPO with inventory_delta semantics and lighter model budget.
        "split_eval_recovery_inventory": [
            "--mode", "backtest_3way",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "180000",
            "--eval-freq", "6000",
            "--learning-rate", "0.0003",
            "--batch-size", "64",
            "--n-steps", "512",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.04",
            "--sell-redirect-mode", "oldest",
            "--semantic-bootstrap-buy-pct", "0.18",
            "--semantic-bootstrap-penalty", "0.05",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.10",
            "--trade-rate-penalty", "0.30",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "96",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.15",
            "--reward-equity-delta-scale", "80.0",
            "--turnover-penalty-rate", "0.06",
            "--continuous-drawdown-penalty", "2.5",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "5400",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[256, 128, 64]",
        ],
        # Recovery v2: anti-hold pressure increase based on hold-dominant zero-trade diagnostics.
        "split_eval_recovery_inventory_v2": [
            "--mode", "backtest_3way",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "180000",
            "--eval-freq", "6000",
            "--learning-rate", "0.0003",
            "--ent-coef", "0.01",
            "--batch-size", "64",
            "--n-steps", "512",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.04",
            "--sell-redirect-mode", "oldest",
            "--semantic-bootstrap-buy-pct", "0.18",
            "--semantic-bootstrap-penalty", "0.0",
            "--inactivity-penalty", "-2.5",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.10",
            "--trade-rate-penalty", "0.30",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "96",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.15",
            "--reward-equity-delta-scale", "80.0",
            "--turnover-penalty-rate", "0.06",
            "--continuous-drawdown-penalty", "2.5",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "5400",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[256, 128, 64]",
        ],
        # Paired control for reward-shaping sensitivity under the same anti-hold settings.
        "split_eval_recovery_inventory_v2_shaped": [
            "--mode", "backtest_3way",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "180000",
            "--eval-freq", "6000",
            "--learning-rate", "0.0003",
            "--ent-coef", "0.01",
            "--batch-size", "64",
            "--n-steps", "512",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "shaped",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.04",
            "--sell-redirect-mode", "oldest",
            "--semantic-bootstrap-buy-pct", "0.18",
            "--semantic-bootstrap-penalty", "0.0",
            "--inactivity-penalty", "-2.5",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.10",
            "--trade-rate-penalty", "0.30",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "96",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.15",
            "--reward-equity-delta-scale", "80.0",
            "--turnover-penalty-rate", "0.06",
            "--continuous-drawdown-penalty", "2.5",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "5400",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[256, 128, 64]",
        ],
        # Recovery profile B: same mapping/reward settings with recurrent policy on walk-forward splits.
        "split_eval_recovery_temporal_ab": [
            "--mode", "backtest_walk_forward",
            "--algo", "recurrent_ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "180000",
            "--eval-freq", "6000",
            "--learning-rate", "0.0003",
            "--batch-size", "64",
            "--n-steps", "512",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.04",
            "--sell-redirect-mode", "oldest",
            "--semantic-bootstrap-buy-pct", "0.18",
            "--semantic-bootstrap-penalty", "0.05",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.10",
            "--trade-rate-penalty", "0.30",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "96",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.15",
            "--reward-equity-delta-scale", "80.0",
            "--turnover-penalty-rate", "0.06",
            "--continuous-drawdown-penalty", "2.5",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "5400",
            "--fee-randomization-pct", "0.10",
            "--model-arch", "[256, 128, 64]",
            "--walk-forward-folds", "3",
            "--walk-forward-train-ratio", "0.50",
            "--walk-forward-val-ratio", "0.15",
            "--walk-forward-test-ratio", "0.15",
            "--walk-forward-step-ratio", "0.10",
            "--lstm-hidden-size", "128",
            "--lstm-layers", "1",
            "--shared-lstm",
        ],
        # Larger run once first_long is stable.
        "full_long": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "365",
            "--max-symbols", "15",
            "--train-steps", "2500000",
            "--eval-freq", "25000",
            "--learning-rate", "0.00015",
            "--batch-size", "256",
            "--n-steps", "2048",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.12",
            "--sell-redirect-mode", "largest_notional",
            "--min-hold-steps", "3",
            "--trade-cooldown-steps", "1",
            "--max-trades-per-window", "72",
            "--trade-window-steps", "240",
            "--constraint-violation-penalty", "0.1",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.10",
            "--trade-rate-penalty", "0.50",
            "--reward-equity-delta-scale", "120.0",
            "--turnover-penalty-rate", "0.10",
            "--continuous-drawdown-penalty", "4.0",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "9000",
            "--fee-randomization-pct", "0.12",
            "--model-arch", "[256, 128, 64]",
        ],
        "parallel_fast": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "120",
            "--max-symbols", "10",
            "--train-steps", "500000",
            "--eval-freq", "10000",
            "--learning-rate", "0.0003",
            "--batch-size", "256",
            "--n-steps", "1024",
            "--n-epochs", "8",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "shaped",
            "--action-mapping-mode", "inventory_delta",
            "--action-deadzone", "0.12",
            "--sell-redirect-mode", "largest_notional",
            "--min-hold-steps", "2",
            "--trade-cooldown-steps", "1",
            "--trade-rate-window", "240",
            "--target-trade-rate", "0.15",
            "--trade-rate-penalty", "0.20",
            "--trade-action-bonus", "10.0",
            "--inactivity-penalty", "-2.0",
            "--randomize-episode-start",
            "--min-episode-steps", "1800",
            "--max-episode-steps", "5400",
            "--model-arch", "[256, 128, 64]",
        ],
        "refine_small": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "60",
            "--max-symbols", "5",
            "--train-steps", "200000",
            "--eval-freq", "10000",
            "--learning-rate", "0.0003",
            "--batch-size", "128",
            "--n-steps", "1024",
            "--n-epochs", "5",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "shaped",
            "--trade-action-bonus", "2.0",
            "--inactivity-penalty", "-0.1",
            "--action-mapping-mode", "inventory_delta",
            "--randomize-episode-start",
            "--model-arch", "[256, 128, 64]",
        ],
        "equity_delta_small": [
            "--mode", "train",
            "--algo", "ppo",
            "--device", "cuda",
            "--days", "60",
            "--max-symbols", "5",
            "--train-steps", "200000",
            "--eval-freq", "10000",
            "--learning-rate", "0.0003",
            "--batch-size", "128",
            "--n-steps", "1024",
            "--n-epochs", "5",
            "--fee-regime", "mexc_spot_mx",
            "--reward-mode", "equity_delta",
            "--reward-equity-delta-scale", "100.0",
            "--num-envs", "2",
            "--randomize-episode-start",
            "--model-arch", "[256, 128, 64]",
        ],
    }
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}")
    return list(profiles[profile])


def _override_arg(argv: list[str], key: str, value: str) -> list[str]:
    if key in argv:
        idx = argv.index(key)
        if idx + 1 < len(argv):
            argv[idx + 1] = str(value)
            return argv
    argv.extend([key, str(value)])
    return argv


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "rl_trader.py"] + _profile_args(args.profile)

    if args.train_steps is not None:
        _override_arg(cmd, "--train-steps", str(args.train_steps))
    if args.days is not None:
        _override_arg(cmd, "--days", str(args.days))
    if args.max_symbols is not None:
        _override_arg(cmd, "--max-symbols", str(args.max_symbols))
    if args.num_envs is not None:
        _override_arg(cmd, "--num-envs", str(args.num_envs))
    if args.device:
        _override_arg(cmd, "--device", args.device)

    if args.dashboard:
        state_file = args.dashboard_state or f"rl_dashboard_state_{args.profile}.json"
        cmd.extend([
            "--dashboard",
            "--dashboard-port", str(args.dashboard_port),
            "--dashboard-state", state_file,
        ])

    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_with_tee(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch longer RL training runs with curated profiles.")
    parser.add_argument(
        "--profile",
        choices=[
            "smoke",
            "first_long",
            "first_long_recovery",
            "split_eval_recovery_inventory",
            "split_eval_recovery_inventory_v2",
            "split_eval_recovery_inventory_v2_shaped",
            "split_eval_recovery_temporal_ab",
            "full_long",
            "parallel_fast",
            "refine_small",
            "equity_delta_small",
        ],
        default="split_eval_recovery_inventory",
    )
    parser.add_argument("--train-steps", type=int, default=None, help="Override train steps")
    parser.add_argument("--days", type=int, default=None, help="Override days")
    parser.add_argument("--max-symbols", type=int, default=None, help="Override symbol cap")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard updates and streamlit server")
    parser.add_argument("--dashboard-port", type=int, default=8766)
    parser.add_argument("--dashboard-state", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
    parser.add_argument("--notes", type=str, default="", help="Free-text note stored with run artifacts")
    parser.add_argument(
        "--objective",
        choices=["non_zero_eval_trades", "trade_band_target", "return_floor_with_trade_min"],
        default="non_zero_eval_trades",
        help="Primary recovery objective tag for this run",
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to rl_trader.py")
    args = parser.parse_args()

    cmd = _build_command(args)
    experiment_tags = _derive_experiment_tags(cmd, args.profile, args.objective)
    if args.profile != "smoke" and not args.dashboard:
        print(
            "[HEAVY_RUN] Note: dashboard is recommended for longer/larger runs. "
            "Use --dashboard for full telemetry and split diagnostics."
        )
    print("[HEAVY_RUN] Command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    if args.dashboard:
        state_path = Path(args.dashboard_state or f"rl_dashboard_state_{args.profile}.json").resolve()
        print(f"[HEAVY_RUN] Dashboard state file: {state_path}")
        print(f"[HEAVY_RUN] Dashboard URL: http://localhost:{args.dashboard_port}")

    if args.dry_run:
        return

    stamp = _utc_ts()
    run_name = f"{stamp}_{args.profile}"
    artifact_root = Path("artifacts") / "experiments"
    launch_root = artifact_root / "launches"
    log_root = artifact_root / "logs"
    summary_root = artifact_root / "summaries"
    dashboard_state_path = args.dashboard_state or f"rl_dashboard_state_{args.profile}.json"

    manifest_path = launch_root / f"{run_name}.json"
    run_log_path = log_root / f"{run_name}.log"

    _write_json(
        manifest_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "notes": args.notes,
            "command": cmd,
            "experiment_tags": experiment_tags,
            "dashboard": {
                "enabled": bool(args.dashboard),
                "port": int(args.dashboard_port),
                "state_file": str(Path(dashboard_state_path).resolve()),
            },
            "overrides": {
                "train_steps": args.train_steps,
                "days": args.days,
                "max_symbols": args.max_symbols,
                "device": args.device,
            },
        },
    )
    print(f"[HEAVY_RUN] Launch manifest: {manifest_path}")
    print(f"[HEAVY_RUN] Console log: {run_log_path}")

    start = time.time()
    return_code = _run_with_tee(cmd, run_log_path)
    elapsed = time.time() - start

    summary_payload = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "return_code": int(return_code),
        "elapsed_seconds": round(float(elapsed), 3),
        "log_file": str(run_log_path),
        "experiment_tags": experiment_tags,
    }
    if args.dashboard:
        state_path = Path(dashboard_state_path)
        try:
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            run_meta = state_data.get("run", {}) if isinstance(state_data, dict) else {}
            summary_payload["dashboard"] = {
                "state_file": str(state_path.resolve()),
                "run_id": run_meta.get("run_id"),
                "status": run_meta.get("status"),
                "final_summary": run_meta.get("final_summary"),
            }
        except Exception:
            summary_payload["dashboard"] = {
                "state_file": str(state_path.resolve()),
                "parse_error": True,
            }

    summary_path = summary_root / f"{run_name}.json"
    _write_json(summary_path, summary_payload)
    print(f"[HEAVY_RUN] Run summary: {summary_path}")

    if return_code != 0:
        print(f"[HEAVY_RUN] Training failed with code {return_code}")
        sys.exit(return_code)

    print(f"[HEAVY_RUN] Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()


