"""Can the walk-forward signals pay Kraken's costs if they trade less?

Kraken (the EU venue since 2026-10-27) charges 0.15–0.40% maker and
0.30–0.80% taker at retail volumes, plus a median EUR spread of ~10 bp. The
daily top-10% portfolio of signal_walk_forward.py turns over ~1.3× per day,
which at these costs loses ~1.5–4 per year. This script tests portfolio
constructions that trade less, on the walk-forward predictions saved with
`signal_walk_forward.py --save-predictions` (out of sample: every
prediction comes from a model fit on earlier data only):

- signal smoothing: an EMA (half-life in days) of each coin's daily
  cross-sectional z-score, so one noisy day does not swap the holdings
- slower rebalancing: every 1, 3 or 7 days
- an exit band: buy the top --top-fraction, but keep a held coin until it falls out of
  the top `exit` fraction. Kept coins are not traded back to equal weight;
  only the cash of the coins sold is spent on the new ones.
- a weight cap (--weight-cap, a multiple of the equal weight): a winner
  that drifts above it is trimmed, and the excess goes to the other
  holdings, so one coin cannot dominate the result
- more holdings (--top-fraction, default 25% ≈ 14 coins): the top 10% are
  only ~5 coins, and the result then hangs on one or two of them

The universe is restricted to the coins Kraken lists in EUR today
(scripts/kraken_costs.py; a coin's listing date on Kraken is not checked).
Costs per unit traded, from the strategy's own volume tier at --capital-eur:

- taker: the taker fee plus half of the coin's median Kraken spread (market
  orders; the conservative case)
- maker: the maker fee and no spread (limit orders that all fill at the
  limit; optimistic, since missed fills and adverse selection are ignored)

Each variant is a long-only portfolio compared with the equal-weight basket
of the same universe (no fees for the basket). Many variants are tried on the
same data, so the best one is biased upward. The table therefore also shows
the net result in each half of the period: a variant chosen on the first half
should hold up in the second.

The final report runs one frozen variant (--variant) on the sealed holdout:
--include-holdout reads the full coin panel, and --start sets the first
rebalance (the EMA warms up on the days before it). The grid search is not
allowed there.

Usage:
    .conda/bin/python scripts/kraken_backtest.py \
        --predictions h1=preds_h1.npz h7=preds_h7.npz
    .conda/bin/python scripts/kraken_backtest.py --include-holdout \
        --start 2025-02-18T18:40 --predictions h7=preds_h7_full.npz \
        --models ridge --variant 2 7 0.5
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coin_panel
import kraken_costs
from spike_economics import DECISION_MINUTES, EVALUATION_START, MINUTES_PER_YEAR

MODELS = ["ridge", "mlp"]
SMOOTHING_HALF_LIVES_DAYS = [0, 2, 5]
REBALANCE_DAYS = [1, 3, 7]
EXIT_FRACTIONS = [0.1, 0.25, 0.5]


def daily_zscore(signal: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row, over the tradable coins with a value.

    Model outputs change scale at each refit; z-scores keep the EMA across
    refits meaningful.
    """
    masked = np.where(tradable & np.isfinite(signal), signal, np.nan)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        mean = np.nanmean(masked, axis=1, keepdims=True)
        std = np.nanstd(masked, axis=1, keepdims=True)
    return (masked - mean) / (std + 1e-12)


def ema(values: np.ndarray, half_life: float) -> np.ndarray:
    """EMA along rows; restarts at a coin's first finite value after a gap."""
    if half_life == 0:
        return values
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    out = np.full_like(values, np.nan)
    state = np.full(values.shape[1], np.nan)
    for i, row in enumerate(values):
        fresh = np.isnan(state)
        state = np.where(fresh, row, alpha * row + (1 - alpha) * state)
        out[i] = state
    return out


def select_weights(
    signal: np.ndarray,
    coins: np.ndarray,
    drifted: np.ndarray,
    top_fraction: float,
    exit_fraction: float,
    weight_cap: float = 0.0,
    min_holdings: int = 3,
    max_holdings: int | None = None,
) -> np.ndarray:
    """New portfolio weights at a rebalance (shared with paper_trade.py).

    ``coins`` are the eligible coin indices, ``drifted`` the current weights
    (their sum falls short of 1 by the cash). Buys the top ``top_fraction`` of
    the eligible coins by ``signal``, but at most ``max_holdings`` (a small
    account can't split into many orders above the exchange minimum); keeps a
    held coin while it ranks within the top ``exit_fraction``, leaves kept
    coins at their drifted weight and splits the cash of the sold coins over
    the new ones. ``weight_cap`` (a multiple of the equal weight) trims
    winners that drifted above it and gives the excess to the other holdings.
    """
    n_coins = len(drifted)
    ranked = coins[np.argsort(-signal[coins])]  # best first
    n_holdings = max(min_holdings, round(top_fraction * len(coins)))
    if max_holdings is not None:
        n_holdings = max(1, min(n_holdings, max_holdings))
    keep_limit = max(n_holdings, round(exit_fraction * len(coins)))
    held = set(np.where(drifted > 0)[0])
    kept = [c for c in ranked[:keep_limit] if c in held][:n_holdings]
    new = [c for c in ranked if c not in kept][: n_holdings - len(kept)]

    weights = np.zeros(n_coins)
    weights[kept] = drifted[kept]
    freed = 1.0 - weights.sum()
    if new:
        weights[new] = freed / len(new)
    else:
        weights /= weights.sum()
    if weight_cap > 0:
        cap = weight_cap / n_holdings
        for _ in range(10):
            over = weights > cap + 1e-12
            if not over.any():
                break
            excess = (weights[over] - cap).sum()
            weights[over] = cap
            room = (weights > 0) & (weights < cap)
            if not room.any():
                break
            weights[room] += excess / room.sum()
    return weights


def apply_min_orders(
    target: np.ndarray, drifted: np.ndarray, value_eur: float, min_order_eur: float
) -> np.ndarray:
    """The weights reached when orders below the exchange minimum are skipped.

    As paper_trade.rebalance trades: sells first, where a trim below the
    minimum is skipped but a whole position always sells (its units cleared
    the minimum when bought). The buys then share the cash, scaled down if it
    falls short; a buy below the minimum is skipped and its cash stays idle.
    """
    min_weight = min_order_eur / value_eur
    weights = drifted.copy()
    sell = (target < drifted) & ((drifted - target >= min_weight) | (target == 0))
    weights[sell] = target[sell]
    wanted = np.clip(target - drifted, 0, None)
    cash = 1.0 - weights.sum()
    scale = min(1.0, cash / wanted.sum()) if wanted.sum() > 0 else 0.0
    adds = wanted * scale
    adds[adds < min_weight] = 0.0
    return weights + adds


def banded_backtest(
    signal: np.ndarray,
    period_returns: np.ndarray,
    eligible_rows: np.ndarray,
    top_fraction: float,
    exit_fraction: float,
    weight_cap: float = 0.0,
    min_holdings: int = 3,
    capital_eur: float = 0.0,
    min_order_eur: float = 0.0,
    min_position_eur: float = 0.0,
) -> dict:
    """Long the top coins with an exit band, vs the equal-weight basket.

    All arrays have one row per rebalance. Returns per-period gross excess
    log returns and the traded weight per coin (|Δw|, for the costs).

    With ``capital_eur``, a small account's limits apply as in paper trading:
    at most value / ``min_position_eur`` holdings, and no orders below
    ``min_order_eur`` (`apply_min_orders`). The value follows the gross
    return.
    """
    n_periods, n_coins = signal.shape
    drifted = np.zeros(n_coins)
    value_eur = capital_eur
    small_account = bool(capital_eur and (min_order_eur or min_position_eur))
    gross, traded = np.zeros(n_periods), np.zeros((n_periods, n_coins))
    basket_log = np.zeros(n_periods)
    active = np.zeros(n_periods, dtype=bool)
    for p in range(n_periods):
        eligible = (
            eligible_rows[p] & np.isfinite(signal[p]) & np.isfinite(period_returns[p])
        )
        coins = np.where(eligible)[0]
        if len(coins) < 2 * min_holdings:
            continue
        active[p] = True
        max_holdings = (
            int(value_eur // min_position_eur)
            if capital_eur and min_position_eur
            else None
        )
        weights = select_weights(
            signal[p],
            coins,
            drifted,
            top_fraction,
            exit_fraction,
            weight_cap,
            min_holdings,
            max_holdings,
        )
        if small_account and min_order_eur:
            weights = apply_min_orders(weights, drifted, value_eur, min_order_eur)
        traded[p] = np.abs(weights - drifted)

        returns = np.nan_to_num(period_returns[p])
        basket = np.zeros(n_coins)
        basket[coins] = 1.0 / len(coins)
        basket_log[p] = np.log1p(basket @ returns)
        # Weights short of 1 are cash, which earns nothing
        growth = 1.0 + weights @ returns
        gross[p] = np.log(growth) - basket_log[p]
        grown = weights * (1.0 + returns)
        # A small account keeps that cash; otherwise the drifted weights are
        # renormalized, as in the frozen backtest (the cap can leave a
        # shortfall when no holding has room)
        drifted = grown / (growth if small_account else grown.sum())
        value_eur *= growth
    return {
        "gross": gross[active],
        "basket": basket_log[active],
        "traded": traded[active],
        "active": active,
    }


def net_of_costs(
    result: dict,
    years: float,
    half_spread: np.ndarray,
    capital_eur: float,
    eur_usd: float,
) -> dict:
    """Charge the fee tier implied by the strategy's own volume, two ways."""
    turnover_per_day = result["traded"].sum() / (years * 365)
    monthly_volume_usd = turnover_per_day * 30 * capital_eur * eur_usd
    maker_fee, taker_fee = kraken_costs.fee_rates(monthly_volume_usd)
    out = {"turnover_per_day": turnover_per_day, "taker_fee": taker_fee}
    for name, cost_per_unit in (
        ("taker", taker_fee + half_spread),
        ("maker", np.full_like(half_spread, maker_fee)),
    ):
        cost = -np.log1p(-(result["traded"] @ np.nan_to_num(cost_per_unit)))
        out[name] = result["gross"] - cost
    return out


def t_stat(series: np.ndarray) -> float:
    return float(series.mean() / (series.std(ddof=1) + 1e-12) * np.sqrt(len(series)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="name=path pairs of signal_walk_forward.py --save-predictions files",
    )
    parser.add_argument("--capital-eur", type=float, default=10_000)
    parser.add_argument(
        "--min-order-eur",
        type=float,
        default=0.0,
        help="Skip orders below this (Kraken's minimums are ~€4-5); 0 = off",
    )
    parser.add_argument(
        "--min-position-eur",
        type=float,
        default=0.0,
        help="Hold at most value / this many coins; 0 = no limit",
    )
    parser.add_argument(
        "--eur-usd", type=float, default=1.11, help="Kraken EUR/USD, 2026-10-28"
    )
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=1.5,
        help="Max weight as a multiple of the equal weight; 0 = no cap",
    )
    parser.add_argument("--show", type=int, default=25, help="Rows of the table")
    parser.add_argument("--out", default=None, help="Optional CSV of all variants")
    parser.add_argument("--placebos", type=int, default=200, help="0 to skip")
    parser.add_argument("--placebo-variants", type=int, default=5)
    parser.add_argument(
        "--models", nargs="+", default=MODELS, choices=MODELS, help="Signals to use"
    )
    parser.add_argument(
        "--variant",
        nargs=3,
        type=float,
        default=None,
        metavar=("HALF_LIFE", "REBALANCE_DAYS", "EXIT"),
        help="Run only this variant instead of the grid",
    )
    parser.add_argument(
        "--start", default=None, help="First rebalance (default: 2023-01-16)"
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Use the full panel through the sealed holdout (final report only; "
        "requires --variant)",
    )
    args = parser.parse_args()
    if args.include_holdout and args.variant is None:
        parser.error("--include-holdout runs one frozen --variant, not the grid")
    if args.variant is not None:
        half_lives, rebalance_days, exit_fractions = (
            [args.variant[0]],
            [int(args.variant[1])],
            [args.variant[2]],
        )
    else:
        half_lives, rebalance_days, exit_fractions = (
            SMOOTHING_HALF_LIVES_DAYS,
            REBALANCE_DAYS,
            EXIT_FRACTIONS,
        )
    start = np.datetime64(args.start or EVALUATION_START, "m")

    panel = coin_panel.load_panel(include_holdout=args.include_holdout)
    symbols = list(panel["symbols"])
    costs = kraken_costs.load_costs().set_index("symbol")
    on_kraken = np.array([s in costs.index for s in symbols])
    half_spread = np.array(
        [costs.spread_bp.get(s, np.nan) / 2e4 for s in symbols], dtype=np.float64
    )
    tradable = panel["tradable"] & on_kraken
    times = panel["decision_times"]
    first_row = int(np.searchsorted(times, EVALUATION_START))
    day_rows = np.arange(first_row, len(times), coin_panel.ROWS_PER_DAY)
    print(
        f"{on_kraken.sum()} of {len(symbols)} coins on Kraken; median "
        f"{np.median(tradable[day_rows].sum(1)):.0f} tradable per day; "
        f"rebalancing from {start} to {times[-1]}; capital "
        f"€{args.capital_eur:,.0f}",
        flush=True,
    )

    signals = {}
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        with np.load(path, allow_pickle=True) as f:
            assert list(f["symbols"]) == symbols, f"{path}: different coins"
            assert np.array_equal(f["decision_times"], times), (
                f"{path}: different times"
            )
            for model in args.models:
                signals[f"{name}_{model}"] = daily_zscore(
                    f[model][day_rows].astype(np.float64), tradable[day_rows]
                )

    holding = {
        days: coin_panel.holding_returns(panel["close"], days * coin_panel.ROWS_PER_DAY)
        for days in rebalance_days
    }

    def run(zscores, half_life, days, exit_fraction):
        """One variant: the backtest and the times of its rebalances."""
        smoothed = ema(zscores, half_life)
        # Rebalance every `days` days; the last one needs a full holding period
        steps = np.arange(np.searchsorted(times[day_rows], start), len(day_rows), days)
        steps = steps[day_rows[steps] + days * coin_panel.ROWS_PER_DAY < len(times)]
        rebalance_rows = day_rows[steps]
        result = banded_backtest(
            smoothed[steps],
            holding[days][rebalance_rows],
            tradable[rebalance_rows],
            args.top_fraction,
            exit_fraction,
            args.weight_cap,
            capital_eur=args.capital_eur,
            min_order_eur=args.min_order_eur,
            min_position_eur=args.min_position_eur,
        )
        result["times"] = times[rebalance_rows[result["active"]]]
        result["per_year"] = MINUTES_PER_YEAR / (
            days * coin_panel.ROWS_PER_DAY * DECISION_MINUTES
        )
        return result

    rows = []
    for (name, zscores), half_life, days, exit_fraction in itertools.product(
        signals.items(), half_lives, rebalance_days, exit_fractions
    ):
        if exit_fraction < args.top_fraction:
            continue
        result = run(zscores, half_life, days, exit_fraction)
        per_year = result["per_year"]
        years = len(result["gross"]) / per_year
        net = net_of_costs(result, years, half_spread, args.capital_eur, args.eur_usd)
        # Halves of the evaluated period, to see whether a result holds in both
        first_half = np.arange(len(result["times"])) < len(result["times"]) // 2
        rows.append(
            {
                "signal": name,
                "half_life": half_life,
                "rebalance": days,
                "exit": exit_fraction,
                "turnover/day": net["turnover_per_day"],
                "taker_fee": net["taker_fee"],
                "gross/yr": result["gross"].mean() * per_year,
                "gross_t": t_stat(result["gross"]),
                "taker/yr": net["taker"].mean() * per_year,
                "taker_t": t_stat(net["taker"]),
                "taker_h1": net["taker"][first_half].mean() * per_year,
                "taker_h2": net["taker"][~first_half].mean() * per_year,
                "maker/yr": net["maker"].mean() * per_year,
                "maker_t": t_stat(net["maker"]),
                "basket/yr": result["basket"].mean() * per_year,
                "portfolio/yr": (result["basket"] + net["taker"]).mean() * per_year,
            }
        )

    table = pd.DataFrame(rows).sort_values("taker/yr", ascending=False)
    if args.out:
        table.to_csv(args.out, index=False)
    print(
        f"\n{len(table)} variants; excess log return over the Kraken basket per "
        "year. taker = taker fee + half spread; maker = maker fee only; h1/h2 = "
        "taker net in each half of the period; basket and portfolio (net of taker "
        "costs) are absolute log returns per year.\n"
    )
    print(
        table.head(args.show).to_string(
            index=False,
            float_format=lambda v: f"{v:+.3f}" if abs(v) < 10 else f"{v:.0f}",
        )
    )
    # How the cost model moves the best variant per signal
    print("\nBest variant per signal (by taker net):")
    print(
        table.groupby("signal", sort=False)
        .head(1)
        .to_string(index=False, float_format=lambda v: f"{v:+.3f}")
    )

    # Placebo: the same construction on random signals with the real
    # signal's day-to-day persistence. Holding drifted weights against a
    # re-equalized basket earns or loses money without any signal; the
    # placebo gross shows how much, and how often chance beats the signal.
    if args.placebos:
        print(
            f"\nPlacebo check of the top {args.placebo_variants} variants "
            f"({args.placebos} random AR(1) signals each; gross/yr):"
        )
        rng = np.random.default_rng(0)
        for _, row in table.head(args.placebo_variants).iterrows():
            zscores = signals[row["signal"]]
            smoothed = ema(zscores, row["half_life"])
            finite = np.isfinite(smoothed[1:]) & np.isfinite(smoothed[:-1])
            rho = np.corrcoef(smoothed[1:][finite], smoothed[:-1][finite])[0, 1]
            placebo_gross, placebo_turnover = [], []
            for _ in range(args.placebos):
                noise = rng.standard_normal(zscores.shape)
                for i in range(1, len(noise)):
                    noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho**2) * noise[i]
                noise[~np.isfinite(zscores)] = np.nan
                result = run(noise, 0, row["rebalance"], row["exit"])
                placebo_gross.append(result["gross"].mean() * result["per_year"])
                years = len(result["gross"]) / result["per_year"]
                placebo_turnover.append(result["traded"].sum() / (years * 365))
            placebo_gross = np.array(placebo_gross)
            print(
                f"  {row['signal']} hl {row['half_life']} rebal {row['rebalance']}d "
                f"exit {row['exit']:.2f}: real {row['gross/yr']:+.3f} | placebo "
                f"mean {placebo_gross.mean():+.3f}, sd {placebo_gross.std():.3f}, "
                f"share ≥ real {np.mean(placebo_gross >= row['gross/yr']):.1%} | "
                f"turnover real {row['turnover/day']:.3f}, placebo "
                f"{np.mean(placebo_turnover):.3f}/day (rho {rho:.3f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
