"""Compute finance metrics from a saved Parquet action log."""

import glob
import numpy as np
import pandas as pd


def eval_report():
    f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
    df = pd.read_parquet(f)
    pv_series = df["portfolio"].to_numpy()
    returns = np.diff(pv_series) / np.array(pv_series[:-1])
    sharpe = (
        returns.mean() / (returns.std() + 1e-8) * np.sqrt(1440)
    )  # annualised at 1-min
    downside = returns[returns < 0]
    sortino = returns.mean() / (downside.std() + 1e-8) * np.sqrt(1440)
    peak = np.maximum.accumulate(pv_series)
    dd = (np.array(pv_series) - peak) / peak
    print(
        f"Sharpe={sharpe:.2f}  Sortino={sortino:.2f}  MaxDD={dd.min() * 100:.1f}%  FinalPV={pv_series[-1]:.2f}"
    )
    traded_symbols: set[str] = set(df["symbol"].tolist())
    print(
        f"Symbols traded counter: {len(traded_symbols)} , including: {sorted(traded_symbols)}"
    )


if __name__ == "__main__":
    eval_report()
