"""Score the rule-based reference strategies on the validation slice (ROADMAP P2-2).

Runs each strategy in `crypto_rl.experiment.reference_strategies` through the
same env, fees, rebalance band and decision interval as the agent, over the
full validation slice of the reward ablation (scripts/reward_ablation.py), and
compares it with the passive baselines over the same bars. No training is
involved, and the test slice is never touched. Results go to
logs/reference_strategies/val_fee<fee>.json.

Run from the repo root once the ablation caches exist
(`python scripts/reward_ablation.py build`):
    python scripts/reference_strategies.py
    python scripts/reference_strategies.py --fee-rate 0   # edge before fees
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from reward_ablation import LOGGER, REPO, base_config, load_slices

sys.path.insert(0, str(REPO))

from crypto_rl.experiment.evaluation_utils import BASELINE_NAMES
from crypto_rl.experiment.fold_runner import _eval_env_args, run_reference_strategies

OUT = REPO / "logs" / "reference_strategies"


def summarize(result: dict, decision_interval: int) -> dict:
    pv0 = result["initial_portfolio_value"]
    # One mark per bar; the last is replaced by the liquidation value
    pvs = np.array(result["env_portfolio_values"], dtype=float)
    pvs[-1] = result["final_portfolio_value"]
    pvs = np.concatenate([[pv0], pvs])
    max_dd = float(np.max(1.0 - pvs / np.maximum.accumulate(pvs)))
    return {
        "return": result["final_portfolio_value"] / pv0 - 1.0,
        "excess": result["excess_returns"],
        "max_drawdown": max_dd,
        "mean_exposure": result["mean_exposure"],
        "fills": int(result["trades_count"]),
        "fees_pct": float(result["fees_paid"]) / pv0,
        "decisions": result["steps"],
        "bars": result["bars"],
        "decision_interval": decision_interval,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fee-rate", type=float, default=None)
    p.add_argument("--decision-interval", type=int, default=15)
    args = p.parse_args()

    overrides = {"decision_interval": args.decision_interval}
    if args.fee_rate is not None:
        overrides["fee_rate"] = args.fee_rate
    cfg = base_config(**overrides)
    _, val, split = load_slices(cfg)
    # parquet_path=None: reset() must use the given slice (see CLAUDE.md)
    env_cfg = dataclasses.replace(cfg, parquet_path=None, n_rows=0)
    env_args = _eval_env_args(
        val, run_id="reference", is_fast_eval=False, config=env_cfg, logger=LOGGER
    )
    results = run_reference_strategies(env_args, show_progress=True)

    first = next(iter(results.values()))
    pv0 = first["initial_portfolio_value"]
    print(
        f"\nValidation slice from {split.train_end_str}, {first['bars']} bars, "
        f"fee {cfg.fee_rate:g}, decision every {cfg.decision_interval} min"
    )
    for name in BASELINE_NAMES:
        ret = first["baselines"][name] / pv0 - 1.0
        print(f"  {name:<17} return {ret * 100:+7.2f}%")
    print(
        f"\n  {'strategy':<17} {'return':>8} {'vs EW':>8} {'vs BTC':>8} "
        f"{'maxDD':>6} {'expo':>5} {'fills':>6} {'fees':>7}"
    )
    summary = {}
    for name, result in results.items():
        s = summarize(result, cfg.decision_interval)
        summary[name] = s
        print(
            f"  {name:<17} {s['return'] * 100:+7.2f}% "
            f"{s['excess']['equal_weight'] * 100:+7.2f}% "
            f"{s['excess']['btc'] * 100:+7.2f}% "
            f"{s['max_drawdown']:6.3f} {s['mean_exposure']:5.2f} "
            f"{s['fills']:6d} {s['fees_pct'] * 100:6.2f}%"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"val_fee{cfg.fee_rate:g}_di{cfg.decision_interval}.json"
    out.write_text(
        json.dumps(
            {
                "fee_rate": cfg.fee_rate,
                "decision_interval": cfg.decision_interval,
                "val_start": split.train_end_str,
                "baseline_returns": {
                    n: first["baselines"][n] / pv0 - 1.0 for n in BASELINE_NAMES
                },
                "strategies": summary,
            },
            indent=2,
        )
    )
    print(f"\nWrote {Path(out).relative_to(REPO)}")


if __name__ == "__main__":
    main()
