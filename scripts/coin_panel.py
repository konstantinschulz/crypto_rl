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
`--build --include-holdout` writes a separate panel through the end of the
dataset, for that final report only; research code loads the cut panel.

Build once (about 15 minutes with 2 worker processes, ~2 GB RAM each):
    .conda/bin/python scripts/coin_panel.py --build
    .conda/bin/python scripts/coin_panel.py --build --include-holdout
"""

import argparse
import dataclasses
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.config import RLConfig
from crypto_rl.env.data_utils import create_temporary_environment_from_long_df
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM

PARQUET_PATH = repo_root / "binance_spot_1m_last4y_single_htf.parquet"
PANEL_PATH = repo_root / "data" / "coin_panel_15m.npz"
# Through the end of the dataset, holdout included: final report only
FULL_PANEL_PATH = repo_root / "data" / "coin_panel_15m_full.npz"
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


def read_symbol(symbol: str, include_holdout: bool = False) -> pd.DataFrame:
    """1-minute bars of one coin (before the holdout), indexed by time."""
    # Row groups hold one coin each; reading just those keeps memory at the
    # coin's size (a filtered dataset scan peaked near 4 GB for one coin)
    parquet = pq.ParquetFile(PARQUET_PATH)
    symbol_column = parquet.schema_arrow.get_field_index("symbol")
    row_groups = [
        i
        for i in range(parquet.metadata.num_row_groups)
        if parquet.metadata.row_group(i).column(symbol_column).statistics.min
        <= symbol
        <= parquet.metadata.row_group(i).column(symbol_column).statistics.max
    ]
    table = parquet.read_row_groups(row_groups, columns=OHLCV_COLUMNS + ["symbol"])
    df = table.filter(pc.equal(table["symbol"], symbol)).drop(["symbol"]).to_pandas()
    df = df.set_index("open_time_dt").sort_index()
    df = df[~df.index.duplicated()]
    return df if include_holdout else df[df.index < HOLDOUT_START]


def coin_features(
    symbol: str,
    minute_grid: pd.DatetimeIndex,
    include_holdout: bool = False,
    read=None,
) -> dict | None:
    """Features of one coin on the global minute grid, sampled every 15 minutes.

    The coin is paired with BTC because `precalculate_static_obs` derives
    BTC-relative features (relative momentum, BTC correlation) and the macro
    block from it. ``read(symbol)`` returns a coin's 1-minute bars (default:
    from the dataset parquet; paper trading passes live bars).
    """
    if read is None:

        def read(s):
            return read_symbol(s, include_holdout)

    coin = read(symbol)
    if len(coin) < 2 * 1440:
        return None
    # The coin's own listed span on the global grid; gaps inside it are
    # forward-filled, as the pipeline's pivot does
    span = minute_grid[(minute_grid >= coin.index[0]) & (minute_grid <= coin.index[-1])]
    frames = {symbol: coin.reindex(span).ffill()}
    if symbol != "BTCUSDT":
        frames["BTCUSDT"] = read("BTCUSDT").reindex(span).ffill().bfill()
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
    symbol, minute_grid, include_holdout = args
    return symbol, coin_features(symbol, minute_grid, include_holdout)


def assemble_panel(
    symbols: list[str], minute_grid: pd.DatetimeIndex, coin_outputs
) -> dict:
    """The panel dict (as `load_panel` returns it) from `coin_features` outputs.

    ``coin_outputs`` yields (symbol, output or None) pairs.
    """
    decision_times = minute_grid[::DECISION_MINUTES]
    n_times, n_coins = len(decision_times), len(symbols)
    close = np.full((n_times, n_coins), np.nan, np.float32)
    features = np.full((n_times, n_coins, STATIC_PER_ASSET_DIM), np.nan, np.float32)
    lag_returns = np.full((n_times, n_coins, len(RETURN_LAGS)), np.nan, np.float32)
    norm_vol = np.full((n_times, n_coins), np.nan, np.float32)
    tradable = np.zeros((n_times, n_coins), bool)
    macro = np.full((n_times, MACRO_DIM), np.nan, np.float32)

    for symbol, out in coin_outputs:
        j = symbols.index(symbol)
        if out is None:
            print(f"{symbol}: too little data, skipped", flush=True)
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

    return {
        "decision_times": decision_times.tz_convert(None).values.astype(
            "datetime64[m]"
        ),
        "symbols": np.array(symbols),
        "close": close,
        "features": features,
        "lag_returns": lag_returns,
        "norm_vol": norm_vol,
        "tradable": tradable,
        "macro": macro,
        "feature_names": np.array(STATIC_NAMES),
        "lag_names": np.array([f"ret_{lag}" for lag in RETURN_LAGS]),
        "macro_names": np.array(MACRO_NAMES),
    }


def build(workers: int, include_holdout: bool = False) -> None:
    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    symbols = sorted(
        dataset.to_table(columns=["symbol"]).column("symbol").unique().to_pylist()
    )
    # BTC's minute timestamps define the global grid (it is complete from the
    # dataset's first day; one 81-minute gap is kept as is)
    minute_grid = read_symbol("BTCUSDT", include_holdout).index
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(s, minute_grid, include_holdout) for s in symbols]
        panel = assemble_panel(symbols, minute_grid, pool.map(_worker, jobs))

    path = FULL_PANEL_PATH if include_holdout else PANEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **panel)
    n_times, n_coins = panel["close"].shape
    print(f"Saved {path}: {n_times} decision times × {n_coins} coins")


def load_panel(symbols: list[str] | None = None, include_holdout: bool = False) -> dict:
    """Load the panel, optionally restricted to ``symbols``.

    Times are naive UTC ``datetime64[m]``; every time is before HOLDOUT_START
    unless ``include_holdout`` (final report only).
    """
    with np.load(FULL_PANEL_PATH if include_holdout else PANEL_PATH) as f:
        panel = {k: f[k] for k in f.files}
    if not include_holdout:
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


DERIVATIVES_DIR = repo_root / "data" / "binance_derivatives"
METRICS_ROWS_PER_DAY = 288  # 5-minute snapshots
DERIVATIVE_NAMES = [
    "funding_8h",
    "funding_7d",
    "premium_1h",
    "premium_24h",
    "oi_change_1d",
    "oi_change_7d",
    "long_short_ratio",
    "top_trader_ratio",
    "taker_ratio_1h",
    "taker_ratio_1d",
]


def _read_derivatives(
    name: str,
    columns: list[str],
    symbols: list[str],
    end: pd.Timestamp,
    directory: Path = DERIVATIVES_DIR,
) -> pd.DataFrame:
    """One downloaded derivatives table, up to ``end`` (the panel's end)."""
    # Streamed in batches and filtered before keeping: reading the 24M-row
    # metrics table in one go peaked ~4 GB above its ~1 GB result. The symbol
    # stays dictionary-encoded (categorical) rather than one string per row.
    parquet = pq.ParquetFile(directory / f"{name}.parquet", read_dictionary=["symbol"])
    frames = []
    for batch in parquet.iter_batches(
        batch_size=1_000_000, columns=["symbol", "time", *columns]
    ):
        frame = batch.to_pandas()
        # One shared category list, so the concat below stays categorical
        frame["symbol"] = pd.Categorical(frame["symbol"], categories=symbols)
        frames.append(frame[(frame["time"] < end) & frame["symbol"].notna()])
    frame = pd.concat(frames, ignore_index=True)
    del frames
    return frame


def _as_of(
    decision_times: pd.Series, series: pd.Series, max_age: pd.Timedelta
) -> np.ndarray:
    """The latest value of a time-indexed ``series`` at each decision time.

    NaN if that value is older than ``max_age``, so a perpetual that stops
    trading does not leave its last value standing forever.
    """
    right = series.dropna().rename("value").reset_index()
    if right.empty:
        return np.full(len(decision_times), np.nan)
    right["time"] = right["time"].astype("datetime64[ns, UTC]")
    left = pd.DataFrame({"time": decision_times.astype("datetime64[ns, UTC]")})
    merged = pd.merge_asof(
        left, right, on="time", direction="backward", tolerance=max_age
    )
    return merged["value"].to_numpy()


def derivatives_features(
    panel: dict, directory: Path = DERIVATIVES_DIR
) -> tuple[np.ndarray, list[str]]:
    """Perpetual-futures features per coin (times, coins, 10), NaN when absent.

    From scripts/download_binance_derivatives.py (or, for paper trading, the
    same tables built from the live API in ``directory``). Every value is taken as of
    the decision time, using only what was public then: funding when it
    settles, a 15-minute premium bar once it closes, a 5-minute metrics
    snapshot 5 minutes after its timestamp (to be safe about publication).

    - funding_8h / funding_7d: latest settled funding rate, and its 7-day
      mean, scaled to an 8-hour interval (some coins settle every 4 hours)
    - premium_1h / premium_24h: mean perpetual premium over the spot index
    - oi_change_1d / oi_change_7d: log change of open interest (in coins)
    - long_short_ratio / top_trader_ratio: log of the all-account and
      top-trader (by position) long/short ratios
    - taker_ratio_1h / taker_ratio_1d: mean log taker buy/sell volume ratio
    """
    symbols = [str(s) for s in panel["symbols"]]
    times = pd.Series(pd.to_datetime(panel["decision_times"]).tz_localize("UTC"))
    out = np.full((len(times), len(symbols), len(DERIVATIVE_NAMES)), np.nan, np.float32)
    # Nothing after the panel's last decision time is read (for the cut
    # panel, that is the holdout)
    end = times.iloc[-1] + pd.Timedelta(minutes=DECISION_MINUTES)

    funding = _read_derivatives(
        "funding",
        ["funding_interval_hours", "last_funding_rate"],
        symbols,
        end,
        directory,
    )
    premium = _read_derivatives("premium_15m", ["close"], symbols, end, directory)
    premium["time"] += pd.Timedelta(minutes=15)  # usable once the bar closes
    metrics = _read_derivatives(
        "metrics",
        [
            "sum_open_interest",
            "count_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ],
        symbols,
        end,
        directory,
    )
    metrics["time"] += pd.Timedelta(minutes=5)

    # Funding settles every 1–8 hours; snapshots and premium bars every 5–15
    # minutes. Older values mean the perpetual stopped (or paused) trading.
    funding_age, snapshot_age = pd.Timedelta(hours=12), pd.Timedelta(hours=1)

    for j, symbol in enumerate(symbols):
        # Sorted per coin: sorting the 24M-row metrics table at once costs ~1 GB
        f = funding[funding.symbol == symbol].set_index("time").sort_index()
        if len(f):
            rate_8h = f.last_funding_rate * 8.0 / f.funding_interval_hours
            out[:, j, 0] = _as_of(times, rate_8h, funding_age)
            out[:, j, 1] = _as_of(times, rate_8h.rolling("7D").mean(), funding_age)
        p = premium[premium.symbol == symbol].set_index("time")["close"].sort_index()
        if len(p):
            out[:, j, 2] = _as_of(times, p.rolling("1h").mean(), snapshot_age)
            out[:, j, 3] = _as_of(times, p.rolling("24h").mean(), snapshot_age)
        m = (
            metrics[metrics.symbol == symbol]
            .drop(columns="symbol")
            .set_index("time")
            .sort_index()
        )
        if len(m):
            # A regular 5-minute grid, so shifts count snapshots; gaps of up
            # to an hour are bridged, longer ones stay missing
            m = m[~m.index.duplicated()]
            m = m.resample("5min").last().ffill(limit=12)
            log_oi = np.log(m.sum_open_interest.where(m.sum_open_interest > 0))
            positive = m.where(m > 0)
            log_taker = np.log(positive.sum_taker_long_short_vol_ratio)
            snapshot_series = [
                log_oi - log_oi.shift(METRICS_ROWS_PER_DAY),
                log_oi - log_oi.shift(7 * METRICS_ROWS_PER_DAY),
                np.log(positive.count_long_short_ratio),
                np.log(positive.sum_toptrader_long_short_ratio),
                log_taker.rolling(12, min_periods=6).mean(),
                log_taker.rolling(METRICS_ROWS_PER_DAY, min_periods=144).mean(),
            ]
            for k, series in enumerate(snapshot_series, start=4):
                out[:, j, k] = _as_of(times, series, snapshot_age)
    return out, list(DERIVATIVE_NAMES)


def model_features(
    panel: dict, derivatives: tuple[np.ndarray, list[str]] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Feature tensor (times, coins, k) for the supervised probes.

    The 13 per-asset indicators and the vol-normalized lag returns, each also
    demeaned across the tradable coins (the target is relative to the
    basket), plus the 5 BTC macro features broadcast to every coin.
    ``derivatives`` (from `derivatives_features`) adds its features the same
    way, raw and demeaned.
    """
    tradable = panel["tradable"]
    blocks = [panel["features"], panel["lag_returns"]]
    names = list(panel["feature_names"]) + list(panel["lag_names"])
    if derivatives is not None:
        blocks.append(derivatives[0])
        names += derivatives[1]
    # Filled in place, one feature at a time: the full panel's tensor is
    # ~3 GB, and stacking float64 copies would need several times that
    n_times, n_coins = tradable.shape
    n_base = sum(block.shape[2] for block in blocks)
    n_macro = panel["macro"].shape[1]
    features = np.empty((n_times, n_coins, 2 * n_base + n_macro), np.float32)
    k = 0
    for block in blocks:
        for j in range(block.shape[2]):
            features[:, :, k] = block[:, :, j]
            features[:, :, n_base + k] = cross_sectional_demean(
                block[:, :, j], tradable
            )
            k += 1
    features[:, :, 2 * n_base :] = panel["macro"][:, None, :]
    names = names + [f"xs_{n}" for n in names] + list(panel["macro_names"])
    return features, names


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--build", action="store_true", help="Build the panel from the parquet"
    )
    p.add_argument("--workers", type=int, default=2)  # ~2 GB peak RSS each
    p.add_argument(
        "--include-holdout",
        action="store_true",
        help="Build the separate full panel through the holdout (final report)",
    )
    args = p.parse_args()
    if args.build:
        build(args.workers, args.include_holdout)
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
