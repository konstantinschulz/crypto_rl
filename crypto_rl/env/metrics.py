from ale_py import env
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
    returns = np.diff(pv_series) / pv_series[:-1]
    # 1-min annualized return
    annualized_return = returns.mean() * 525600
    # Calculate Max Drawdown
    running_max = np.maximum.accumulate(pv_series)
    drawdowns = (running_max - pv_series) / running_max
    max_drawdown = np.max(drawdowns)
    calmar = annualized_return / max(
        max_drawdown, 1e-8
    )  # returns.mean() / (returns.std() + 1e-8) * np.sqrt(525600)
    return calmar


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
