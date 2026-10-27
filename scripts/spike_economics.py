"""Can the volatility / volume "spike" signals pay their trading costs?

Background (2026-10-26): on the 9-coin setup, two model-free signals had a
consistent positive 24h cross-sectional IC over 762 days:

- intrabar_vol spike: the coin's 1-minute high–low range relative to price,
  minus its own trailing 30-day mean. A coin that is unusually volatile
  right now tends to outperform the basket over the next day.
- rvol_24h spike: the coin's volume relative to its 24h mean, minus its own
  trailing 30-day mean (unusual activity).

An IC is not a profit, though. This script turns each signal into a
long-only portfolio and charges fees on every rebalance. At each rebalance it
holds the top fraction of tradable coins by signal, equal weighted, and
compares that portfolio with the equal-weight basket of all tradable coins
(which is also what the RL reward measures against). It reports gross and
net excess log return per year, turnover, and a t-stat over the rebalance
periods, for several holding periods: a faster rebalance captures more of a
short-lived signal but pays more fees.

Costs: MEXC spot charges 0% maker and 0.05% taker (2026). The bot trades at
decision time with market orders, so 0.05% per unit of turnover is the base
case. --stress-fee-rate adds a run at 0.1% for spread and slippage, which
matter for the smaller coins.

All data comes from the coin panel (scripts/coin_panel.py), which ends at the
sealed fold-3 holdout.

Usage:
    .conda/bin/python scripts/spike_economics.py
    .conda/bin/python scripts/spike_economics.py --symbols BTCUSDT ETHUSDT ...
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coin_panel

# Evaluation starts where the 9-coin walk-forward started (end of the fold-1
# training slice), so the periods are comparable
EVALUATION_START = np.datetime64("2023-01-16T00:00")
HOLDING_PERIODS_DAYS = [1, 3, 7]
MINUTES_PER_YEAR = 1440 * 365
DECISION_MINUTES = coin_panel.DECISION_MINUTES


def spike_signal(panel: dict, feature_name: str) -> np.ndarray:
    """Feature minus its trailing 30-day per-coin mean (no lookahead)."""
    feature_index = list(panel["feature_names"]).index(feature_name)
    feature = panel["features"][:, :, feature_index].astype(np.float64)
    return feature - coin_panel.trailing_coin_mean(feature)


def cross_sectional_zscore(signal: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Standardize across the tradable coins at each time."""
    masked = np.where(tradable, signal, np.nan)
    demeaned = coin_panel.cross_sectional_demean(masked, tradable)
    # Times without tradable coins give an empty std (NaN) by design
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spread = np.nanstd(masked, axis=1, keepdims=True)
        return demeaned / spread


def top_fraction_backtest(
    signal: np.ndarray,
    close: np.ndarray,
    tradable: np.ndarray,
    rebalance_rows: np.ndarray,
    holding_rows: int,
    fee_rate: float,
    top_fraction: float = 0.1,
    min_holdings: int = 3,
) -> dict:
    """Long the top ``top_fraction`` of coins by signal vs the equal-weight basket.

    At each rebalance row the portfolio buys the highest-signal coins among
    those tradable with a finite signal, equal weighted, and holds them for
    ``holding_rows`` (the next rebalance). Turnover is measured against the
    drifted weights of the previous holding, and fees are charged on it. The
    basket is the equal-weight portfolio of the same eligible coins, without
    fees, so the net excess is conservative.

    Returns annualized gross / net excess log returns, turnover per day, and
    t-stats of the per-period excess.
    """
    period_returns = coin_panel.holding_returns(close, holding_rows)
    n_coins = close.shape[1]
    previous_weights_drifted = np.zeros(n_coins)
    gross_excess, net_excess = [], []
    total_turnover = 0.0
    for row in rebalance_rows:
        coin_returns = period_returns[row]
        eligible = tradable[row] & np.isfinite(signal[row]) & np.isfinite(coin_returns)
        eligible_coins = np.where(eligible)[0]
        if len(eligible_coins) < 2 * min_holdings:
            continue
        n_holdings = max(min_holdings, round(top_fraction * len(eligible_coins)))
        ranked = eligible_coins[np.argsort(signal[row, eligible_coins])]
        weights = np.zeros(n_coins)
        weights[ranked[-n_holdings:]] = 1.0 / n_holdings
        basket_weights = np.zeros(n_coins)
        basket_weights[eligible_coins] = 1.0 / len(eligible_coins)

        turnover = np.abs(weights - previous_weights_drifted).sum()
        total_turnover += turnover
        returns = np.nan_to_num(coin_returns)
        portfolio_log_return = np.log1p(weights @ returns)
        basket_log_return = np.log1p(basket_weights @ returns)
        fee_log_cost = -np.log1p(-fee_rate * turnover)
        gross_excess.append(portfolio_log_return - basket_log_return)
        net_excess.append(portfolio_log_return - fee_log_cost - basket_log_return)

        # Weights drift with prices until the next rebalance
        grown = weights * (1.0 + returns)
        previous_weights_drifted = grown / grown.sum()

    gross_excess, net_excess = np.array(gross_excess), np.array(net_excess)
    years = len(gross_excess) * holding_rows * DECISION_MINUTES / MINUTES_PER_YEAR

    def t_stat(series):
        return series.mean() / (series.std(ddof=1) + 1e-12) * np.sqrt(len(series))

    return {
        "periods": len(gross_excess),
        "gross_per_year": gross_excess.sum() / years,
        "net_per_year": net_excess.sum() / years,
        "turnover_per_day": total_turnover / (years * 365),
        "gross_t": t_stat(gross_excess),
        "net_t": t_stat(net_excess),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", nargs="*", default=None, help="Default: all coins")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="MEXC taker fee")
    parser.add_argument("--stress-fee-rate", type=float, default=0.001)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    args = parser.parse_args()

    panel = coin_panel.load_panel(args.symbols)
    tradable = panel["tradable"]
    first_row = int(np.searchsorted(panel["decision_times"], EVALUATION_START))
    print(
        f"{len(panel['symbols'])} coins, median {np.median(tradable[first_row:].sum(1)):.0f} "
        f"tradable; evaluation {panel['decision_times'][first_row]} → "
        f"{panel['decision_times'][-1]}"
    )

    signals = {
        "intrabar_vol spike": spike_signal(panel, "intrabar_vol"),
        "rvol_24h spike": spike_signal(panel, "rvol_24h"),
    }
    # Both spikes, each standardized across coins first so neither dominates
    signals["combined"] = np.nan_to_num(
        cross_sectional_zscore(signals["intrabar_vol spike"], tradable)
    ) + np.nan_to_num(cross_sectional_zscore(signals["rvol_24h spike"], tradable))
    signals["combined"][~np.isfinite(signals["intrabar_vol spike"])] = np.nan

    for fee_rate in (args.fee_rate, args.stress_fee_rate):
        print(
            f"\nTop {args.top_fraction:.0%} vs equal-weight basket, fee {fee_rate:.3%}:"
        )
        for name, signal in signals.items():
            for days in HOLDING_PERIODS_DAYS:
                holding_rows = days * coin_panel.ROWS_PER_DAY
                rebalance_rows = np.arange(
                    first_row, len(tradable) - holding_rows, holding_rows
                )
                r = top_fraction_backtest(
                    signal,
                    panel["close"],
                    tradable,
                    rebalance_rows,
                    holding_rows,
                    fee_rate,
                    args.top_fraction,
                )
                print(
                    f"  {name:<19} hold {days}d: gross {r['gross_per_year']:+.3f}/yr "
                    f"(t {r['gross_t']:+.2f}) | net {r['net_per_year']:+.3f}/yr "
                    f"(t {r['net_t']:+.2f}) | turnover {r['turnover_per_day']:.2f}x/day "
                    f"| {r['periods']} periods"
                )


if __name__ == "__main__":
    main()
