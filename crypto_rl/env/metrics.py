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
    calmar = annualized_return / max(max_drawdown, 1e-8)  # returns.mean() / (returns.std() + 1e-8) * np.sqrt(525600)
    return calmar
