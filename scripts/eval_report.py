"""Compute finance metrics from a saved Parquet action log."""

import glob
import json
import numpy as np
import pandas as pd


def eval_report():
    f2 = sorted(glob.glob("logs/run-*/state.json"))[-1]
    with open(f2) as f3:
        state = json.load(f3)
        keys = ["technical", "finance", "explainability"]
        for key in keys:
            if key in state:
                print(f"{key}: {json.dumps(state[key], indent=2)}")

    f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
    df = pd.read_parquet(f)
    pv_series = df["portfolio"].to_numpy()

    # Safe returns calculation (prevents division by zero when portfolio nears 0)
    denom = pv_series[:-1]
    returns = np.divide(
        np.diff(pv_series),
        denom,
        out=np.zeros_like(np.diff(pv_series), dtype=float),
        where=denom > 1e-8,
    )

    # 1-min annualized return
    annualized_return = returns.mean() * 525600
    # Calculate Max Drawdown
    running_max = np.maximum.accumulate(pv_series)
    drawdowns = (running_max - pv_series) / running_max
    max_drawdown = np.max(drawdowns)
    calmar = annualized_return / max(max_drawdown, 1e-8)

    downside = returns[returns < 0]
    sortino = 0.0
    # 1. Check for insufficient data (Degrees of freedom <= 0)
    if len(returns) > 1 and len(downside) > 1:
        # 2. Check for zero variance (Invalid value encountered in divide)
        std_dev = np.std(returns, ddof=1)
        if std_dev >= 1e-8:
            sortino = returns.mean() / (downside.std() + 1e-8) * np.sqrt(1440)
    peak = np.maximum.accumulate(pv_series)

    # Safe drawdown calculation
    dd = np.divide(
        np.array(pv_series) - peak,
        peak,
        out=np.zeros_like(pv_series, dtype=float),
        where=peak > 1e-8,
    )

    print(
        f"Calmar={calmar:.2f}  Sortino={sortino:.2f}  MaxDD={dd.min() * 100:.1f}%  FinalPV={pv_series[-1]:.2f}"
    )
    traded_symbols: set[str] = set(df["symbol"].tolist())
    print(
        f"Symbols traded counter: {len(traded_symbols)} , including: {sorted(traded_symbols)}"
    )


if __name__ == "__main__":
    eval_report()
