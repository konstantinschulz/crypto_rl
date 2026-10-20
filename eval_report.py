"""Compute finance metrics from a saved Parquet action log."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.env.logging_utils import LoggerBase, selective_logger
from crypto_rl.env.metrics import get_calmar_from_portfolio_series


def get_drawdown_from_portfolio_series(pv_series: np.ndarray) -> float:
    """Compute the maximum drawdown from a portfolio value series."""
    peak = np.maximum.accumulate(pv_series)
    # Safe drawdown calculation
    dd = np.divide(
        np.array(pv_series) - peak,
        peak,
        out=np.zeros_like(pv_series, dtype=float),
        where=peak > 1e-8,
    )
    return dd.min()


def get_sortino_from_portfolio_series(pv_series: np.ndarray) -> float:
    """Compute the Sortino ratio from a portfolio value series."""
    # Safe returns calculation (prevents division by zero when portfolio nears 0)
    denom = pv_series[:-1]
    returns = np.divide(
        np.diff(pv_series),
        denom,
        out=np.zeros_like(np.diff(pv_series), dtype=float),
        where=denom > 1e-8,
    )
    downside = returns[returns < 0]
    sortino = 0.0
    # 1. Check for insufficient data (Degrees of freedom <= 0)
    if len(returns) > 1 and len(downside) > 1:
        # 2. Check for zero variance (Invalid value encountered in divide)
        std_dev = np.std(returns, ddof=1)
        if std_dev >= 1e-8:
            sortino = returns.mean() / (downside.std() + 1e-8) * np.sqrt(1440)
    return sortino


def report_action_log(
    logger: LoggerBase, log_path: str | Path, trial: optuna.Trial | None = None
) -> None:
    """Log finance metrics computed from one Parquet action log."""
    df = pd.read_parquet(log_path)
    pv_series = df["portfolio"].to_numpy()
    calmar = get_calmar_from_portfolio_series(pv_series)
    sortino = get_sortino_from_portfolio_series(pv_series)
    dd = get_drawdown_from_portfolio_series(pv_series)
    logger.log(f"Action log: {log_path}", trial)
    logger.log(str(Counter(df["action_type"])), trial)
    logger.log(f"Final portfolio: {df['portfolio'].iloc[-1]}", trial)
    logger.log(
        f"Calmar={calmar:.2f}  Sortino={sortino:.2f}  MaxDD={dd * 100:.1f}%  FinalPV={pv_series[-1]:.2f}",
        trial,
    )
    # Only fills count as trades: HOLDs and no-op orders carry zero units.
    traded_symbols: set[str] = set(df.loc[df["units"] > 0, "symbol"].tolist())
    logger.log(
        f"Symbols traded counter: {len(traded_symbols)} , including: {sorted(traded_symbols)}",
        trial,
    )


def eval_report(
    logger: LoggerBase,
    config: RLConfig,
    trial: optuna.Trial | None = None,
    fold_results: list[dict] | None = None,
    cv_mean_pv: float | None = None,
    cv_mean_calmar: float | None = None,
    state_file: Path | None = None,
) -> None:
    if trial is None and state_file is not None and state_file.exists():
        with open(state_file) as f3:
            state = json.load(f3)
            keys = ["technical", "finance", "explainability"]
            for key in keys:
                if key in state:
                    dump_value = state[key]
                    if key == "finance":
                        for fold_item in dump_value["evaluation_results"]["cv_folds"]:
                            # remove portfolio values from the fold results to avoid logging large arrays
                            del fold_item["portfolio_values"]
                    logger.log(f"{key}: {json.dumps(dump_value, indent=2)}", trial)
    if trial is not None and fold_results is not None:
        traded_coins: set[str] = set()
        sortino_values = []
        drawdown_values = []
        total_action_types = Counter()
        for fold_result in fold_results:
            total_action_types.update(fold_result["action_type_counts"])
            traded_coins.update(fold_result["traded_coins"])
            pv_series = np.array(fold_result["portfolio_values"])
            sortino_values.append(get_sortino_from_portfolio_series(pv_series))
            drawdown_values.append(get_drawdown_from_portfolio_series(pv_series))
        cv_mean_sortino = float(np.mean(sortino_values))
        cv_mean_maxdd = float(np.mean(drawdown_values))
        logger.log(str(total_action_types), trial)
        logger.log(
            f"Calmar={cv_mean_calmar:.2f}  Sortino={cv_mean_sortino:.2f}  MaxDD={cv_mean_maxdd * 100:.1f}%  FinalPV={cv_mean_pv:.2f}",
            trial,
        )
        logger.log(
            f"Coins traded counter: {len(traded_coins)} , including: {sorted(traded_coins)}",
            trial,
        )
        logger.log(json.dumps(config.to_dict(), indent=2), trial)
        return
    # Report each fold's final-test log explicitly; globbing for the latest
    # file picks up intermediate eval episodes or other runs' logs.
    for fold_result in fold_results or []:
        log_path = fold_result.get("action_log_path")
        if log_path is None or not Path(log_path).exists():
            logger.log(
                f"Fold {fold_result.get('fold')}: no final-test action log found.",
                trial,
            )
            continue
        logger.log(f"Fold {fold_result.get('fold')} ({fold_result.get('split')}):", trial)
        report_action_log(logger, log_path, trial)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <actions_eval_*.parquet>")
    with selective_logger("") as logger:
        report_action_log(logger, sys.argv[1])
