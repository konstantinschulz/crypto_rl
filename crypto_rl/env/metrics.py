import numpy as np


def calculate_calmar_ratio(portfolio_values: list[dict]) -> float:
    """
    Calculate the Calmar ratio for a given series of portfolio values.

    Parameters
    ----------
    portfolio_values : list[dict]
        A list of dictionaries containing portfolio values at each timestep.

    Returns
    -------
    float
        The Calmar ratio.
    """
    pv_series = np.array([v["value"] for v in portfolio_values])
    return get_calmar_from_portfolio_series(pv_series)


def get_calmar_from_portfolio_series(pv_series: np.ndarray) -> float:
    """
    Calculate a Return-Scaled Calmar Ratio (Return^2 / Drawdown).
    """
    if len(pv_series) < 2:
        return 0.0
    total_return = (pv_series[-1] - pv_series[0]) / pv_series[0]
    annualized_return = total_return * (525600 / len(pv_series))
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
        unrealized_pnl = (
            float(mce.holdings[i] * (last_prices[i] - mce.avg_entry_price[i]))
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
            "fees_paid": float(mce.per_asset_fees[i]),
            "current_holdings": float(mce.holdings[i]),
        }
    return summary
