import numpy as np


def calculate_calmar_ratio(
    portfolio_values: list[dict], minutes_per_point: int = 1
) -> float:
    """
    Calculate the Calmar ratio for a given series of portfolio values.

    Parameters
    ----------
    portfolio_values : list[dict]
        A list of dictionaries containing portfolio values at each timestep.
    minutes_per_point : int
        Minutes between consecutive values, e.g. ``config.decision_interval``
        for a series recorded once per env step. Used to annualize.

    Returns
    -------
    float
        The Calmar ratio.
    """
    pv_series = np.array([v["value"] for v in portfolio_values])
    return get_calmar_from_portfolio_series(pv_series, minutes_per_point)


def get_calmar_from_portfolio_series(
    pv_series: np.ndarray, minutes_per_point: int = 1
) -> float:
    """
    Calculate a Return-Scaled Calmar Ratio (Return^2 / Drawdown).
    """
    if len(pv_series) < 2:
        return 0.0
    total_return = (pv_series[-1] - pv_series[0]) / pv_series[0]
    annualized_return = total_return * (525600 / (len(pv_series) * minutes_per_point))
    running_max = np.maximum.accumulate(pv_series)
    drawdowns = (running_max - pv_series) / running_max
    max_drawdown = np.max(drawdowns)
    # 1% Floor to prevent zero-drawdown division explosions
    safe_max_drawdown = max(max_drawdown, 0.01)
    base_calmar = annualized_return / safe_max_drawdown
    # If the bot loses money, return the raw negative Calmar to punish drawdowns
    if annualized_return <= 0:
        return float(base_calmar)
    # FIX: Scale by the RAW return (total_return) instead of annualized_return.
    # A 5% raw return (0.05) will appropriately scale down the metric, keeping the score in the single or double digits regardless of the time window.
    adjusted_score = base_calmar * total_return
    return float(adjusted_score)


def max_drawdown(pv_series: np.ndarray) -> float:
    """Largest peak-to-trough loss of a PV series, as a positive fraction."""
    pv_series = np.asarray(pv_series, dtype=float)
    if len(pv_series) < 2:
        return 0.0
    peaks = np.maximum.accumulate(pv_series)
    drawdowns = np.divide(
        peaks - pv_series, peaks, out=np.zeros_like(pv_series), where=peaks > 1e-8
    )
    return float(drawdowns.max())


def selection_score(
    excess_log_return: float, max_dd: float, drawdown_coef: float
) -> float:
    """Model-selection objective: ``excess_log_return − λ·maxDD`` (ROADMAP P2-3).

    Unlike the Calmar variant above, it is monotonic in both terms, continuous
    at zero return, and not annualized, so short windows are not blown up.
    """
    return float(excess_log_return - drawdown_coef * max_dd)


def episode_metrics(
    initial_pv: float,
    final_pv: float,
    bar_pvs: list[float] | np.ndarray,
    equal_weight_final_pv: float,
    turnover: float,
    n_bars: int,
    drawdown_coef: float,
) -> dict[str, float]:
    """Summary metrics of one evaluation episode.

    ``bar_pvs`` is the env's mark-to-market PV at every held one-minute bar.
    ``final_pv`` is the liquidation value (exit fee charged) and
    ``equal_weight_final_pv`` the equal-weight buy-and-hold baseline over the
    same bars and fees. ``turnover`` is the traded notional summed over the
    episode, each trade as a fraction of the pre-trade PV.
    """
    pv_series = np.concatenate(([initial_pv], np.asarray(bar_pvs, float), [final_pv]))
    log_return = float(np.log(max(final_pv, 1e-8) / initial_pv))
    excess = float(np.log(max(final_pv, 1e-8) / max(equal_weight_final_pv, 1e-8)))
    max_dd = max_drawdown(pv_series)
    days = max(n_bars, 1) / 1440.0
    return {
        "return": float(final_pv / initial_pv - 1.0),
        "log_return": log_return,
        "excess_log_return": excess,
        "max_drawdown": max_dd,
        "turnover": float(turnover),
        "turnover_per_day": float(turnover / days),
        "score": selection_score(excess, max_dd, drawdown_coef),
    }


def summarize_folds(
    fold_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Mean, sample std, min and max of each metric across folds."""
    if not fold_metrics:
        return {}
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def get_per_asset_summary(env) -> dict[str, dict[str, float]]:
    """Returns detailed evaluation metrics broken down by asset symbol."""
    from crypto_rl.env.minimal_env import MinimalCryptoEnv

    mce: MinimalCryptoEnv = env
    last_prices = (
        mce.prices_arr[mce.current_step - 1]
        if mce.current_step > 0
        else mce.prices_arr[0]
    )
    summary = {}
    for i, sym in enumerate(mce.asset_names):
        realized_pnl = float(mce.per_asset_realized_pnl[i])
        # Open positions are valued net of the exit fee, so realized plus
        # unrealized PnL sums to the episode's liquidation-value PnL.
        exit_fee = float(mce.holdings[i] * last_prices[i] * mce.fee_rate)
        unrealized_pnl = (
            float(mce.holdings[i] * last_prices[i] - exit_fee)
            - float(mce.holdings[i] * mce.avg_entry_price[i])
            if mce.holdings[i] > 1e-8
            else 0.0
        )
        trades = int(mce.per_asset_trades[i])
        wins = int(mce.per_asset_wins[i])
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        summary[sym] = {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "trades": trades,
            "wins": wins,
            "win_rate_pct": win_rate,
            "fees_paid": float(mce.per_asset_fees[i]) + exit_fee,
            "exit_fee": exit_fee,
            "current_holdings": float(mce.holdings[i]),
        }
    return summary
