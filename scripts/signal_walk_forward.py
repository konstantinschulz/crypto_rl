"""Monthly walk-forward check of the 24h cross-sectional signal.

`signal_probe.py` found a 24h cross-sectional IC of about 0.07 on the three
69-day validation windows of the 9-coin setup, mostly from a static asset
ranking. This script re-tests the question on consecutive 30-day windows from
2023-01-16 up to the sealed fold-3 holdout (2025-02-18), on the coin panel
(scripts/coin_panel.py): by default every coin in the dataset, or a subset
via --symbols. The windows cover the fold 1–2 test slices of the 9-coin
layout, which were unsealed for this research on 2026-10-26 (CLAUDE.md).

Models are refit every --refit-days on all earlier rows (expanding window,
purged by the 24h horizon, subsampled to hourly rows for speed) and predict
the following period. The target is a coin's 24h (or --horizon-days) forward
simple return minus
the mean over the coins tradable at that time (--target log and --loss huber
reproduce the earlier, median-leaning setup). Reported over the pooled daily
IC series (one rank IC across coins per day):

- model ICs: raw, the static part (each coin's mean prediction over the 30
  days before the refit, held constant: a fixed coin ranking) and the
  time-varying part (prediction minus its trailing 30-day per-coin mean)
- model-free "spike" features: a feature minus its trailing 30-day per-coin
  mean
- per-year means, the share of positive months, and a month-block bootstrap
  95% CI
- the mean excess return per prediction decile: a high rank IC can come
  from ordering the typical coin while a few large moves set the means
- top-decile vs equal-weight basket portfolios, net of fees (see
  spike_economics.py for the backtest)

With --horizon-days > 1, the daily ICs and decile means overlap in time, so
their t-stats are inflated by up to that factor; the backtest rebalances
once per horizon and has no overlap.

Usage:
    .conda/bin/python scripts/signal_walk_forward.py
    .conda/bin/python scripts/signal_walk_forward.py --symbols BTCUSDT ETHUSDT ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coin_panel
import signal_probe
from spike_economics import EVALUATION_START, spike_signal, top_fraction_backtest

WINDOW_DAYS = 30
STATIC_DAYS = 30  # history for the static / time-varying split
SPIKE_FEATURES = ["intrabar_vol", "rvol_24h", "vol_norm"]
MIN_COINS_PER_DAY = 5


def daily_rank_ic(
    prediction: np.ndarray, target: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    """Spearman correlation across the usable coins, one value per given row.

    Rows with fewer than MIN_COINS_PER_DAY usable coins give NaN.
    """
    from scipy.stats import rankdata  # here, to keep paper trading's imports light

    ics = np.full(len(prediction), np.nan)
    for i in range(len(prediction)):
        ok = usable[i] & np.isfinite(prediction[i]) & np.isfinite(target[i])
        if ok.sum() < MIN_COINS_PER_DAY:
            continue
        prediction_rank = rankdata(prediction[i, ok])
        target_rank = rankdata(target[i, ok])
        ics[i] = np.corrcoef(prediction_rank, target_rank)[0, 1]
    return ics


def summarize(
    ic_by_day: np.ndarray, month_of_day: np.ndarray, year_of_day: np.ndarray
) -> dict:
    """Mean IC with t-stat, month-block bootstrap CI, per-year means."""
    keep = np.isfinite(ic_by_day)
    ic_by_day, month_of_day, year_of_day = (
        ic_by_day[keep],
        month_of_day[keep],
        year_of_day[keep],
    )
    mean = ic_by_day.mean()
    t = mean / (ic_by_day.std(ddof=1) + 1e-12) * np.sqrt(len(ic_by_day))
    # Resample whole months, keeping each month's daily ICs together, so
    # serial correlation within a month does not shrink the interval
    months = np.unique(month_of_day)
    ics_per_month = [ic_by_day[month_of_day == m] for m in months]
    rng = np.random.default_rng(0)
    bootstrap_means = [
        np.concatenate(
            [ics_per_month[i] for i in rng.integers(0, len(months), len(months))]
        ).mean()
        for _ in range(2000)
    ]
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "ic": float(mean),
        "t": float(t),
        "ci": (float(low), float(high)),
        "per_year": {
            int(y): float(ic_by_day[year_of_day == y].mean())
            for y in np.unique(year_of_day)
        },
        "positive_months": float(np.mean([m.mean() > 0 for m in ics_per_month])),
        "days": len(ic_by_day),
    }


def decile_mean_excess(
    prediction: np.ndarray, excess: np.ndarray, usable: np.ndarray
) -> tuple[list[float], float]:
    """Mean excess return per prediction decile (bp), and the D10 − D1 t-stat.

    The mean, not the rank, is what an equal-weight holding of a decile
    earns. A model can order the typical coin well (high rank IC) while the
    decile means stay flat, if large moves of a few coins dominate them.
    """
    per_decile = [[] for _ in range(10)]
    spread = []
    for i in range(len(prediction)):
        ok = usable[i] & np.isfinite(prediction[i])
        if ok.sum() < 20:
            continue
        ranks = np.argsort(np.argsort(prediction[i, ok]))
        decile = (10 * ranks) // ok.sum()
        means = [excess[i, ok][decile == k].mean() for k in range(10)]
        for k in range(10):
            per_decile[k].append(means[k])
        spread.append(means[9] - means[0])
    spread = np.array(spread)
    spread_t = spread.mean() / (spread.std(ddof=1) + 1e-12) * np.sqrt(len(spread))
    return [1e4 * float(np.mean(d)) for d in per_decile], float(spread_t)


def standardization_stats(train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std of the training rows, ignoring NaN (float32).

    Column by column: nanmean/nanstd copy the whole ~2M-row matrix.
    """
    n_features = train_rows.shape[1]
    mean, std = np.zeros(n_features, np.float32), np.zeros(n_features, np.float32)
    for j in range(n_features):
        column = train_rows[:, j].astype(np.float64)
        mean[j] = np.nanmean(column)
        std[j] = np.nanstd(column) + 1e-8
    return mean, std


def standardize(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z-score, clipped to ±5; NaN → 0 (the mean). float32 throughout, since
    float64 temporaries of the training matrix cost ~1 GB of peak memory."""
    z = rows.astype(np.float32, copy=True)
    z -= mean
    z /= std
    np.clip(z, -5, 5, out=z)
    return np.nan_to_num(z, copy=False)


def standardize_with(train_rows: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """z-score with the training rows' statistics (see `standardize`)."""
    mean, std = standardization_stats(train_rows)
    return [standardize(a, mean, std) for a in (train_rows, *others)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", nargs="*", default=None, help="Default: all coins")
    parser.add_argument("--seeds", type=int, default=2, help="MLP seeds, averaged")
    parser.add_argument("--refit-days", type=int, default=90)
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="Forward-return horizon of the target, in days",
    )
    parser.add_argument(
        "--train-row-stride",
        type=int,
        default=4,
        help="Use every n-th 15-minute row for training (4 = hourly)",
    )
    parser.add_argument(
        "--target",
        choices=["simple", "log"],
        default="simple",
        help="Train on simple excess returns (what a portfolio earns) or log "
        "excess returns (closer to the median; used before 2026-10-26)",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
        help="MLP loss: mse targets the mean, huber leans toward the median "
        "(used before 2026-10-26)",
    )
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="MEXC taker fee")
    parser.add_argument(
        "--out", default=None, help="Optional JSON file for the results"
    )
    parser.add_argument(
        "--derivatives",
        action="store_true",
        help="Add the Binance perpetual-futures features (funding, premium, "
        "open interest, long/short and taker ratios; see coin_panel."
        "derivatives_features) to the models, and report each on its own",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Optional .npz file for the model predictions (times × coins)",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Continue the walk-forward through the sealed fold-3 holdout (the "
        "full panel). Only for the one final report of a frozen setup",
    )
    args = parser.parse_args()
    horizon_rows = args.horizon_days * coin_panel.ROWS_PER_DAY

    panel = coin_panel.load_panel(args.symbols, args.include_holdout)
    times = panel["decision_times"]
    tradable = panel["tradable"]
    n_times, n_coins = tradable.shape
    derivatives = coin_panel.derivatives_features(panel) if args.derivatives else None
    features, _ = coin_panel.model_features(panel, derivatives)
    n_features = features.shape[2]
    forward_log = coin_panel.forward_log_returns(panel["close"], horizon_rows)
    has_forward = tradable & np.isfinite(forward_log)
    # What a portfolio earns: the simple return over the basket's mean. Rank
    # ICs are the same for both targets; the model fit and the decile means
    # are not.
    simple_excess = coin_panel.cross_sectional_demean(
        np.expm1(forward_log), has_forward
    )
    if args.target == "simple":
        target = simple_excess
    else:
        target = coin_panel.cross_sectional_demean(forward_log, has_forward)
    # A (time, coin) sample counts if the coin is tradable and has a target
    usable = tradable & np.isfinite(target)
    print(
        f"{n_coins} coins, median {np.median(tradable.sum(1)):.0f} tradable, "
        f"{n_features} features",
        flush=True,
    )

    predictions = {
        name: np.full((n_times, n_coins), np.nan)
        for name in ("ridge", "mlp", "mlp_static", "mlp_varying")
    }
    first_row = int(np.searchsorted(times, EVALUATION_START))
    refit_rows = args.refit_days * coin_panel.ROWS_PER_DAY
    static_rows = STATIC_DAYS * coin_panel.ROWS_PER_DAY
    for period_start in range(first_row, n_times, refit_rows):
        period_end = min(period_start + refit_rows, n_times)
        # Purge: training targets must end before the period starts
        train_times = np.arange(0, period_start - horizon_rows, args.train_row_stride)
        train_mask = usable[train_times]
        x_train = features[train_times][train_mask]
        y_train = target[train_times][train_mask]
        # Trim the most extreme 0.5% on each side for training only: a few
        # listing pumps of several hundred percent would dominate the MSE fit
        low, high = np.percentile(y_train, [0.5, 99.5])
        y_train = np.clip(y_train, low, high)
        # Predict the period plus the 30 days before it, which the static /
        # time-varying split uses as its reference
        context_start = max(0, period_start - static_rows)
        x_context = features[context_start:period_end].reshape(-1, n_features)
        x_train, x_context = standardize_with(x_train, x_context)
        # Rows are time-major, so the inner early-stopping split stays
        # chronological; the gap (in rows) keeps overlapping targets apart
        coins_per_time = int(np.median(train_mask.sum(1)))
        gap = (horizon_rows // args.train_row_stride + 1) * coins_per_time
        ridge_weights = signal_probe.fit_ridge(x_train, y_train, gap)
        ridge_context = signal_probe.ridge_predict(x_context, ridge_weights).reshape(
            -1, n_coins
        )
        mlp_context = np.zeros_like(ridge_context)
        for seed in range(args.seeds):
            mlp = signal_probe.fit_mlp(x_train, y_train, gap, seed, loss=args.loss)
            mlp_context += mlp(x_context).reshape(-1, n_coins) / args.seeds

        offset = period_start - context_start
        predictions["ridge"][period_start:period_end] = ridge_context[offset:]
        predictions["mlp"][period_start:period_end] = mlp_context[offset:]
        # Static part: each coin's mean prediction over the 30 days before the
        # period, held constant. Time-varying part: prediction minus its
        # trailing 30-day per-coin mean.
        context_tradable = tradable[context_start:period_end]
        tradable_predictions = np.where(context_tradable, mlp_context, np.nan)
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            static = np.nanmean(tradable_predictions[:offset], axis=0)
        predictions["mlp_static"][period_start:period_end] = static
        trailing = coin_panel.trailing_coin_mean(tradable_predictions, days=STATIC_DAYS)
        predictions["mlp_varying"][period_start:period_end] = (mlp_context - trailing)[
            offset:
        ]

        period_days = slice(period_start, period_end, coin_panel.ROWS_PER_DAY)
        period_ic = np.nanmean(
            daily_rank_ic(
                predictions["mlp"][period_days],
                target[period_days],
                usable[period_days],
            )
        )
        print(
            f"refit for {times[period_start]} → {times[period_end - 1]}: "
            f"{len(y_train)} training rows, MLP IC {period_ic:+.3f}",
            flush=True,
        )

    del features  # ~3 GB on the full panel; only the refits need it
    spikes = {f"spike_{name}": spike_signal(panel, name) for name in SPIKE_FEATURES}
    signals = {**predictions, **spikes}
    if derivatives is not None:
        # Each derivatives feature as a signal of its own, model-free
        for k, name in enumerate(derivatives[1]):
            signals[f"deriv_{name}"] = derivatives[0][:, :, k].astype(np.float64)

    day_rows = np.arange(first_row, n_times, coin_panel.ROWS_PER_DAY)
    month_of_day = (day_rows - first_row) // (WINDOW_DAYS * coin_panel.ROWS_PER_DAY)
    year_of_day = times[day_rows].astype("datetime64[Y]").astype(int) + 1970
    results = {}
    print(
        f"\n{args.horizon_days}-day cross-sectional rank IC, {len(day_rows)} days in "
        f"{month_of_day.max() + 1} windows of {WINDOW_DAYS} days"
    )
    for name, signal in signals.items():
        ic_by_day = daily_rank_ic(signal[day_rows], target[day_rows], usable[day_rows])
        summary = summarize(ic_by_day, month_of_day, year_of_day)
        results[name] = summary
        per_year = " ".join(f"{y}: {v:+.3f}" for y, v in summary["per_year"].items())
        print(
            f"  {name:<24} IC {summary['ic']:+.4f} (t {summary['t']:+5.2f}, 95% CI "
            f"{summary['ci'][0]:+.3f}…{summary['ci'][1]:+.3f}) | months > 0: "
            f"{summary['positive_months']:.0%} | {per_year}"
        )

    print(
        f"\nMean {args.horizon_days}-day simple excess return over the basket by prediction "
        "decile (bp; D1 = lowest), and top-minus-bottom decile t-stat:"
    )
    for name in ["mlp", "mlp_varying", "ridge"] + [
        n for n in signals if n.startswith("deriv_")
    ]:
        decile_means, spread_t = decile_mean_excess(
            signals[name][day_rows], simple_excess[day_rows], usable[day_rows]
        )
        results[name]["decile_mean_excess_bp"] = decile_means
        print(
            f"  {name:<24} "
            + " ".join(f"{m:+6.1f}" for m in decile_means)
            + f" | D10−D1 t {spread_t:+.2f}"
        )

    print(
        f"\nTop 10% vs equal-weight basket, rebalanced every {args.horizon_days} day(s), "
        f"fee {args.fee_rate:.3%}:"
    )
    for name in (
        "mlp",
        "mlp_static",
        "mlp_varying",
        "ridge",
        "spike_intrabar_vol",
        "spike_rvol_24h",
    ):
        backtest = top_fraction_backtest(
            signals[name],
            panel["close"],
            tradable,
            day_rows[: -args.horizon_days : args.horizon_days],
            horizon_rows,
            args.fee_rate,
        )
        results[name]["top10"] = backtest
        print(
            f"  {name:<20} gross {backtest['gross_per_year']:+.2f}/yr "
            f"(t {backtest['gross_t']:+.2f}) | net {backtest['net_per_year']:+.2f}/yr "
            f"(t {backtest['net_t']:+.2f}) | turnover {backtest['turnover_per_day']:.2f}x/day"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)
    if args.save_predictions:
        np.savez(
            args.save_predictions,
            decision_times=times,
            symbols=panel["symbols"],
            **{name: p.astype(np.float32) for name, p in predictions.items()},
        )


if __name__ == "__main__":
    main()
