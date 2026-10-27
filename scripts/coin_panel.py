"""A 15-minute panel of every coin in the dataset, for signal research.

The RL pipeline needs every coin listed for the whole window and builds a
(rows × 5 + 13·coins) static matrix in RAM, so it stays at 9 coins. Signal
research needs breadth instead. This module turns the 1-minute parquet into a
compact panel with one row per 15-minute decision time and one column per
coin (71 coins), with NaN where a coin is not listed.

Per coin it uses the repo's own feature code
(`create_temporary_environment_from_long_df`, i.e. `precalculate_static_obs`)
on that coin plus BTC, so the 13 per-asset indicators and the normalized
volatility match what the agent observes. It also stores vol-normalized log
returns over the observation's lags (1 minute to 24 hours).

The panel is cut at HOLDOUT_START, the start of the fold-3 test slice of the
3-fold layout; that slice stays sealed for the final report (CLAUDE.md).

Build once (about 15 minutes with 2 worker processes, ~5 GB RAM each):
    .conda/bin/python scripts/coin_panel.py --build
"""

import argparse
import dataclasses
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.config import RLConfig
from crypto_rl.env.data_utils import create_temporary_environment_from_long_df
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM

PARQUET_PATH = repo_root / "binance_spot_1m_last4y_single_htf.parquet"
PANEL_PATH = repo_root / "data" / "coin_panel_15m.npz"
# First bar of the fold-3 test slice (18M rows, 3 folds): sealed holdout
HOLDOUT_START = pd.Timestamp("2025-02-18 18:40", tz="UTC")
DECISION_MINUTES = 15
# Observation window in bars, as in the runs (feature windows depend on it)
WINDOW_SIZE = 240
# Lags (in 1-minute bars) of the vol-normalized past returns
RETURN_LAGS = [1, 5, 15, 30, 60, 120, 240, 720, 1440]
# A coin counts as tradable this long after listing: the 24h features and
# 30-day trailing means need history, and launch days are not tradable signal
TRADABLE_AFTER_LISTING = pd.Timedelta(days=7)
STATIC_NAMES = [
    "vol_norm",
    "intrabar_vol",
    "vwap_dev",
    "mom_norm",
    "rsi",
    "macd",
    "rel_mom_norm",
    "rvol_24h",
    "vwap_dev_24h",
    "btc_corr_60m",
    "htf_slope_15m",
    "htf_slope_1h",
    "htf_regime_24h",
]
MACRO_NAMES = ["btc_mom_norm", "btc_vol_norm", "btc_bull", "btc_ranging", "btc_bear"]
OHLCV_COLUMNS = [
    "open_time_dt",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "htf_slope_15m",
    "htf_slope_1h",
    "htf_regime_24h",
]


def read_symbol(symbol: str) -> pd.DataFrame:
    """1-minute bars of one coin before the holdout, indexed by time."""
    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    df = dataset.to_table(
        columns=OHLCV_COLUMNS, filter=ds.field("symbol") == symbol
    ).to_pandas()
    df = df.set_index("open_time_dt").sort_index()
    df = df[~df.index.duplicated()]
    return df[df.index < HOLDOUT_START]


def coin_features(symbol: str, minute_grid: pd.DatetimeIndex) -> dict | None:
    """Features of one coin on the global minute grid, sampled every 15 minutes.

    The coin is paired with BTC because `precalculate_static_obs` derives
    BTC-relative features (relative momentum, BTC correlation) and the macro
    block from it.
    """
    coin = read_symbol(symbol)
    if len(coin) < 2 * 1440:
        return None
    # The coin's own listed span on the global grid; gaps inside it are
    # forward-filled, as the pipeline's pivot does
    span = minute_grid[(minute_grid >= coin.index[0]) & (minute_grid <= coin.index[-1])]
    frames = {symbol: coin.reindex(span).ffill()}
    if symbol != "BTCUSDT":
        frames["BTCUSDT"] = read_symbol("BTCUSDT").reindex(span).ffill().bfill()
    long_df = pd.concat(
        [f.assign(symbol=s, open_time=np.arange(len(span))) for s, f in frames.items()],
        ignore_index=True,
    )
    config = dataclasses.replace(
        RLConfig(), parquet_path=None, n_rows=0, window_size=WINDOW_SIZE
    )
    env, norm_vol, asset_names = create_temporary_environment_from_long_df(
        long_df, config
    )
    col = asset_names.index(symbol)
    n_assets = len(asset_names)

    # Decision rows: every 15th minute of the global grid that lies in the span
    grid_pos = minute_grid.get_indexer(span)
    in_span = np.where(grid_pos % DECISION_MINUTES == 0)[0]
    static = env.precalc_static_obs[in_span]
    per_asset = np.stack(
        [
            static[:, MACRO_DIM + f * n_assets + col]
            for f in range(STATIC_PER_ASSET_DIM)
        ],
        axis=1,
    )
    log_close = np.log(np.maximum(env.prices_arr[:, col].astype(np.float64), 1e-12))
    vol = np.maximum(norm_vol[in_span, col].astype(np.float64), 1e-8)
    lag_returns = np.stack(
        [
            np.where(
                in_span >= lag,
                (log_close[in_span] - log_close[np.maximum(in_span - lag, 0)])
                / (vol * np.sqrt(lag)),
                np.nan,
            )
            for lag in RETURN_LAGS
        ],
        axis=1,
    )
    return {
        "decision_index": grid_pos[in_span] // DECISION_MINUTES,
        "close": env.prices_arr[in_span, col],
        "features": per_asset,
        "lag_returns": lag_returns.astype(np.float32),
        "norm_vol": vol.astype(np.float32),
        "macro": static[:, :MACRO_DIM] if symbol == "BTCUSDT" else None,
        "listed_from": span[0],
    }


def _worker(args):
    symbol, minute_grid = args
    return symbol, coin_features(symbol, minute_grid)


def build(workers: int) -> None:
    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    symbols = sorted(
        dataset.to_table(columns=["symbol"]).column("symbol").unique().to_pylist()
    )
    # BTC's minute timestamps define the global grid (it is complete from the
    # dataset's first day; one 81-minute gap is kept as is)
    minute_grid = read_symbol("BTCUSDT").index
    decision_times = minute_grid[::DECISION_MINUTES]
    n_times, n_coins = len(decision_times), len(symbols)
    close = np.full((n_times, n_coins), np.nan, np.float32)
    features = np.full((n_times, n_coins, STATIC_PER_ASSET_DIM), np.nan, np.float32)
    lag_returns = np.full((n_times, n_coins, len(RETURN_LAGS)), np.nan, np.float32)
    norm_vol = np.full((n_times, n_coins), np.nan, np.float32)
    tradable = np.zeros((n_times, n_coins), bool)
    macro = np.full((n_times, MACRO_DIM), np.nan, np.float32)

    with ProcessPoolExecutor(workers) as pool:
        jobs = [(s, minute_grid) for s in symbols]
        for symbol, out in pool.map(_worker, jobs):
            j = symbols.index(symbol)
            if out is None:
                print(
                    f"{symbol}: too little data before the holdout, skipped", flush=True
                )
                continue
            rows = out["decision_index"]
            close[rows, j] = out["close"]
            features[rows, j] = out["features"]
            lag_returns[rows, j] = out["lag_returns"]
            norm_vol[rows, j] = out["norm_vol"]
            tradable[rows, j] = decision_times[rows] >= (
                out["listed_from"] + TRADABLE_AFTER_LISTING
            )
            if out["macro"] is not None:
                macro[rows] = out["macro"]
            print(
                f"{symbol}: {len(rows)} decision rows from {out['listed_from']:%Y-%m-%d}",
                flush=True,
            )

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PANEL_PATH,
        decision_times=decision_times.tz_convert(None).values.astype("datetime64[m]"),
        symbols=np.array(symbols),
        close=close,
        features=features,
        lag_returns=lag_returns,
        norm_vol=norm_vol,
        tradable=tradable,
        macro=macro,
        feature_names=np.array(STATIC_NAMES),
        lag_names=np.array([f"ret_{lag}" for lag in RETURN_LAGS]),
        macro_names=np.array(MACRO_NAMES),
    )
    print(f"Saved {PANEL_PATH}: {n_times} decision times × {n_coins} coins")


def load_panel(symbols: list[str] | None = None) -> dict:
    """Load the panel, optionally restricted to ``symbols``.

    Times are naive UTC ``datetime64[m]``; every time is before HOLDOUT_START.
    """
    with np.load(PANEL_PATH) as f:
        panel = {k: f[k] for k in f.files}
    holdout = np.datetime64(HOLDOUT_START.tz_convert(None), "m")
    assert panel["decision_times"].max() < holdout, "panel reaches into the holdout"
    if symbols is not None:
        idx = [list(panel["symbols"]).index(s) for s in symbols]
        for key in (
            "symbols",
            "close",
            "features",
            "lag_returns",
            "norm_vol",
            "tradable",
        ):
            panel[key] = panel[key][:, idx] if panel[key].ndim > 1 else panel[key][idx]
    return panel


ROWS_PER_DAY = 1440 // DECISION_MINUTES


def cross_sectional_demean(values: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Subtract, per time, the mean over the coins tradable at that time."""
    masked = np.where(tradable, values, np.nan)
    # Times without any tradable coin give an empty mean (NaN) by design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(masked, axis=1, keepdims=True)
    return values - mean


def trailing_coin_mean(
    values: np.ndarray, days: int = 30, min_days: int = 7
) -> np.ndarray:
    """Per-coin mean of the previous ``days`` (current row excluded).

    NaN until a coin has ``min_days`` of history in the window, so a fresh
    listing has no reference level yet.
    """
    window = days * ROWS_PER_DAY
    frame = pd.DataFrame(values.astype(np.float64))
    mean = frame.rolling(window, min_periods=min_days * ROWS_PER_DAY).mean().shift(1)
    return mean.to_numpy()


def forward_log_returns(close: np.ndarray, horizon_rows: int) -> np.ndarray:
    """log(close[t + h] / close[t]); NaN when either end is missing.

    Used as a prediction target, so a coin that stops trading inside the
    horizon has no target rather than a made-up one.
    """
    out = np.full(close.shape, np.nan)
    log_close = np.log(close.astype(np.float64))
    out[:-horizon_rows] = log_close[horizon_rows:] - log_close[:-horizon_rows]
    return out


def holding_returns(close: np.ndarray, horizon_rows: int) -> np.ndarray:
    """Simple return from t to t + h for a holder, through delistings.

    Prices are forward-filled, so a coin that stops trading is valued at its
    last price (what a holder could at best have sold for).
    """
    filled = pd.DataFrame(close.astype(np.float64)).ffill().to_numpy()
    out = np.full(close.shape, np.nan)
    out[:-horizon_rows] = filled[horizon_rows:] / filled[:-horizon_rows] - 1.0
    return out


def model_features(panel: dict) -> tuple[np.ndarray, list[str]]:
    """Feature tensor (times, coins, k) for the supervised probes.

    The 13 per-asset indicators and the vol-normalized lag returns, each also
    demeaned across the tradable coins (the target is relative to the
    basket), plus the 5 BTC macro features broadcast to every coin.
    """
    tradable = panel["tradable"]
    blocks = [panel["features"], panel["lag_returns"]]
    names = list(panel["feature_names"]) + list(panel["lag_names"])
    base = np.concatenate(blocks, axis=2)
    demeaned = np.stack(
        [cross_sectional_demean(base[:, :, k], tradable) for k in range(base.shape[2])],
        axis=2,
    )
    n_coins = tradable.shape[1]
    macro = np.repeat(panel["macro"][:, None, :], n_coins, axis=1)
    features = np.concatenate([base, demeaned, macro], axis=2).astype(np.float32)
    names = names + [f"xs_{n}" for n in names] + list(panel["macro_names"])
    return features, names


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--build", action="store_true", help="Build the panel from the parquet"
    )
    p.add_argument("--workers", type=int, default=2)  # ~5 GB peak RSS each
    args = p.parse_args()
    if args.build:
        build(args.workers)
    else:
        panel = load_panel()
        tradable = panel["tradable"]
        print(f"{len(panel['decision_times'])} times × {len(panel['symbols'])} coins")
        print(
            f"Tradable coins per time: min {tradable.sum(1).min()}, "
            f"median {np.median(tradable.sum(1)):.0f}, max {tradable.sum(1).max()}"
        )


if __name__ == "__main__":
    main()
