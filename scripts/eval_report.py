"""Compute finance metrics from a saved Parquet action log."""

import glob
import numpy as np
import pandas as pd


def eval_report():
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

    sharpe = (
        returns.mean() / (returns.std() + 1e-8) * np.sqrt(1440)
    )  # annualised at 1-min

    downside = returns[returns < 0]
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
        f"Sharpe={sharpe:.2f}  Sortino={sortino:.2f}  MaxDD={dd.min() * 100:.1f}%  FinalPV={pv_series[-1]:.2f}"
    )
    traded_symbols: set[str] = set(df["symbol"].tolist())
    print(
        f"Symbols traded counter: {len(traded_symbols)} , including: {sorted(traded_symbols)}"
    )


if __name__ == "__main__":
    eval_report()
