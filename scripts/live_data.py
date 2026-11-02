"""Live Binance data in the training format, for paper (and later live) trading.

The model was trained on the 1-minute spot parquet (scripts/coin_panel.py)
and the USDⓈ-M futures archive (scripts/download_binance_derivatives.py).
Both archives lag by a day or more, so the trader reads the same data from
Binance's public REST APIs (no account or key needed):

- spot 1-minute klines from data-api.binance.vision, plus the HTF columns the
  parquet carries (scripts/preprocess_htf_indicators.py)
- futures metrics (open interest, long/short ratios, taker ratio), funding
  and 15-minute premium klines from fapi.binance.com

REST and archive values agree (checked 2026-10-28 on BTC, FET and PEPE for
2026-10-20): open interest exactly, the ratios to REST's 4-decimal rounding.
The archive's metrics time is the REST timestamp minus 5 minutes for open
interest and the long/short ratios, and equal to it for the taker ratio.

Everything is cached under data/live/ and updated incrementally; the
derivatives tables keep the archive's schema, so `coin_panel.
derivatives_features(panel, LIVE_DERIVATIVES_DIR)` reads them unchanged.
`live_panel` then builds the coin panel for the last WINDOW_DAYS with the
training feature code.

Usage:
    .conda/bin/python scripts/live_data.py --update          # refresh the cache
    .conda/bin/python scripts/live_data.py --parity 2026-01-20
The parity check builds a window from the archives with the live code path
and compares its features with the full panel, which saw all history.
"""

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coin_panel
from download_binance_derivatives import PERP_PREFIXES
from preprocess_htf_indicators import compute_htf_indicators

repo_root = Path(__file__).resolve().parent.parent
LIVE_DIR = repo_root / "data" / "live"
KLINES_DIR = LIVE_DIR / "klines"
LIVE_DERIVATIVES_DIR = LIVE_DIR / "derivatives"
# The frozen paper-trading model (scripts/paper_trade.py); a small VM gets this
# file instead of the coin panel
PAPER_MODEL_PATH = repo_root / "data" / "paper" / "model.npz"
SPOT_URL = "https://data-api.binance.vision/api/v3"
FUTURES_URL = "https://fapi.binance.com"
# History behind each decision: the features look back up to 7 days, the
# 24h EMA of the HTF columns needs a few days to forget its start, and the
# trader smooths two weeks of daily predictions
WINDOW_DAYS = 24
CACHE_DAYS = 30  # the futures metrics endpoints serve the last 30 days only
# Decision times sit on the training grid. It started at :07 past each
# quarter hour, but an 81-minute gap in BTC's bars on 2023-03-24 shifted it to
# :12 (:12, :27, :42, :57), and the backtest's daily rows to 01:27 UTC.
GRID_OFFSET_MINUTES = 12
METRIC_ENDPOINTS = {
    # archive column: (endpoint, field, minutes to add to the REST timestamp)
    "sum_open_interest": ("openInterestHist", "sumOpenInterest", -5),
    "count_long_short_ratio": ("globalLongShortAccountRatio", "longShortRatio", -5),
    "sum_toptrader_long_short_ratio": (
        "topLongShortPositionRatio",
        "longShortRatio",
        -5,
    ),
    "sum_taker_long_short_vol_ratio": ("takerlongshortRatio", "buySellRatio", 0),
}


class Client:
    """requests with retries and a pause between calls (public rate limits).

    The futures-data endpoints allow 1000 requests per 5 minutes per IP, hence
    0.3 s; spot klines allow ~3000 per minute, hence much less.
    """

    def __init__(self, pause: float = 0.3, spot_pause: float = 0.05):
        self.session = requests.Session()
        self.pause = pause
        self.spot_pause = spot_pause

    def get(self, url: str, **params):
        for attempt in range(5):
            time.sleep(self.spot_pause if url.startswith(SPOT_URL) else self.pause)
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"{response.status_code}")
                if response.status_code >= 400:
                    # A bad request fails the same way again: no retries
                    response.raise_for_status()
                return response.json()
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else 0
                if attempt == 4 or 400 <= status < 500:
                    raise
                time.sleep(5 * (attempt + 1))
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))


def dataset_symbols() -> list[str]:
    """The coins the model was trained on (its cross-section): from the coin
    panel, or from the paper-trading model where the panel is missing."""
    path = coin_panel.PANEL_PATH
    if not path.exists():
        path = PAPER_MODEL_PATH
    with np.load(path) as f:
        return [str(s) for s in f["symbols"]]


def trading_spot_symbols(client: Client, symbols: list[str]) -> list[str]:
    info = client.get(f"{SPOT_URL}/exchangeInfo", permissions="SPOT")
    trading = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING"}
    return [s for s in symbols if s in trading]


def perpetual_of(client: Client, symbols: list[str]) -> dict[str, str]:
    """Spot symbol → its USDⓈ-M perpetual (BONK trades as 1000BONKUSDT etc.)."""
    info = client.get(f"{FUTURES_URL}/fapi/v1/exchangeInfo")
    perps = {
        s["symbol"]
        for s in info["symbols"]
        if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"
    }
    out = {}
    for spot in symbols:
        match = next((p + spot for p in PERP_PREFIXES if p + spot in perps), None)
        if match:
            out[spot] = match
    return out


def _now_minute() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")


# --- spot klines ---------------------------------------------------------


def update_klines(client: Client, symbol: str, until: pd.Timestamp) -> pd.DataFrame:
    """Fetch the closed 1-minute bars missing from the cache; return the cache."""
    path = KLINES_DIR / f"{symbol}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    start = until - pd.Timedelta(days=CACHE_DAYS)
    if cached is not None and len(cached):
        start = max(start, cached.index[-1] + pd.Timedelta(minutes=1))
    rows = []
    cursor = start
    while cursor < until:
        batch = client.get(
            f"{SPOT_URL}/klines",
            symbol=symbol,
            interval="1m",
            startTime=int(cursor.timestamp() * 1000),
            endTime=int(until.timestamp() * 1000) - 1,
            limit=1000,
        )
        if not batch:
            break
        rows += batch
        cursor = pd.Timestamp(batch[-1][0], unit="ms", tz="UTC") + pd.Timedelta(
            minutes=1
        )
    if rows:
        new = pd.DataFrame(
            [r[:6] for r in rows],
            columns=["open_time", "open", "high", "low", "close", "volume"],
        )
        new.index = pd.to_datetime(new.pop("open_time"), unit="ms", utc=True)
        new.index.name = "open_time_dt"
        new = new.astype(np.float32)
        cached = new if cached is None else pd.concat([cached, new])
    if cached is None:
        return pd.DataFrame()
    cached = cached[~cached.index.duplicated(keep="last")].sort_index()
    cached = cached[cached.index >= until - pd.Timedelta(days=CACHE_DAYS)]
    KLINES_DIR.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(path)
    return cached


def with_htf(bars: pd.DataFrame) -> pd.DataFrame:
    """Add the parquet's HTF columns, computed over the cached bars."""
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    return compute_htf_indicators(out)


# --- futures data --------------------------------------------------------


def _paged(client, url, params, time_key, step_ms, start_ms, end_ms, limit):
    """Rows of a time-paged endpoint between start and end (ms)."""
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = client.get(
            url,
            **params,
            startTime=cursor,
            endTime=min(end_ms, cursor + step_ms * limit) - 1,
            limit=limit,
        )
        rows += batch
        if batch:
            last = batch[-1][time_key] if isinstance(batch[-1], dict) else batch[-1][0]
            cursor = max(int(last) + 1, cursor + step_ms)
            if len(batch) < limit:
                cursor = max(cursor, min(end_ms, cursor + step_ms * limit))
        else:
            cursor += step_ms * limit
    return rows


def _ms(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def fetch_metrics(client, perp, start, end) -> pd.DataFrame:
    """5-minute metrics in the archive schema (time + 4 columns)."""
    columns = []
    for column, (endpoint, field, shift) in METRIC_ENDPOINTS.items():
        rows = _paged(
            client,
            f"{FUTURES_URL}/futures/data/{endpoint}",
            {"symbol": perp, "period": "5m"},
            "timestamp",
            5 * 60_000,
            _ms(start),
            _ms(end),
            500,
        )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        times = pd.to_datetime(frame["timestamp"].astype(np.int64), unit="ms", utc=True)
        series = pd.Series(
            frame[field].astype(float).to_numpy(),
            index=times + pd.Timedelta(minutes=shift),
            name=column,
        )
        # Consecutive pages can repeat their boundary row
        columns.append(series[~series.index.duplicated(keep="last")])
    if not columns:
        return pd.DataFrame()
    metrics = pd.concat(columns, axis=1)
    metrics = metrics[~metrics.index.duplicated()].sort_index()
    return metrics.rename_axis("time").reset_index()


def fetch_funding(client, perp, start, end) -> pd.DataFrame:
    """Funding settlements in the archive schema."""
    rows = _paged(
        client,
        f"{FUTURES_URL}/fapi/v1/fundingRate",
        {"symbol": perp},
        "fundingTime",
        3_600_000,
        _ms(start),
        _ms(end),
        1000,
    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    funding = pd.DataFrame(
        {
            "time": pd.to_datetime(
                frame["fundingTime"].astype(np.int64), unit="ms", utc=True
            ),
            "last_funding_rate": frame["fundingRate"].astype(float),
        }
    ).sort_values("time")
    # The archive states each settlement's interval; REST does not, so take
    # the gap to the previous settlement (rounded to whole hours)
    gaps = funding["time"].diff().dt.total_seconds() / 3600
    funding["funding_interval_hours"] = gaps.round().bfill().fillna(8).clip(1, 8)
    return funding


def fetch_premium(client, perp, start, end) -> pd.DataFrame:
    """15-minute premium-index bars (open time and close), archive schema."""
    rows = _paged(
        client,
        f"{FUTURES_URL}/fapi/v1/premiumIndexKlines",
        {"symbol": perp, "interval": "15m"},
        None,
        15 * 60_000,
        _ms(start),
        _ms(end),
        1500,
    )
    return pd.DataFrame(
        {
            "time": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
            "close": [float(r[4]) for r in rows],
        }
    )


def update_derivatives(
    client: Client, perp_of: dict[str, str], until: pd.Timestamp
) -> list[str]:
    """Refresh the cached futures tables up to ``until`` (incremental).

    Returns the coins whose update failed.
    """
    LIVE_DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    names = ("metrics", "funding", "premium_15m")
    cached = {}
    for name in names:
        path = LIVE_DERIVATIVES_DIR / f"{name}.parquet"
        cached[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    oldest = until - pd.Timedelta(days=CACHE_DAYS) + pd.Timedelta(hours=1)
    new = {name: [] for name in names}
    failed = []
    for spot, perp in perp_of.items():
        # The metrics endpoints refuse start times more than 30 days before
        # now (not before ``until``), so a cold start long after the decision
        # time starts the metrics later, which the window doesn't need. Per
        # coin, since the loop takes a while (or the machine sleeps)
        oldest_metrics = max(
            oldest,
            _now_minute() - pd.Timedelta(days=CACHE_DAYS) + pd.Timedelta(minutes=10),
        )
        starts = {}
        for name in names:
            table = cached[name]
            start = oldest_metrics if name == "metrics" else oldest
            if len(table) and (table.symbol == spot).any():
                # Re-fetch the last 2 hours: the newest rows may have been
                # missing at the previous update
                last = table.loc[table.symbol == spot, "time"].max()
                start = max(start, last - pd.Timedelta(hours=2))
            starts[name] = start
        try:
            tables = {
                "metrics": fetch_metrics(client, perp, starts["metrics"], until),
                # The funding interval is inferred from gaps, so always
                # refetch the whole (small) range
                "funding": fetch_funding(client, perp, oldest, until),
                "premium_15m": fetch_premium(
                    client, perp, starts["premium_15m"], until
                ),
            }
        # One coin's failure (API error, malformed rows) must not stop the rest
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            print(f"{spot}: futures data update failed ({error!r})", flush=True)
            failed.append(spot)
            continue
        for name, frame in tables.items():
            if not frame.empty:
                frame.insert(0, "symbol", spot)
                new[name].append(frame)
    for name in names:
        if name == "funding" and len(cached[name]):
            # Replaced as a whole, except for coins whose update failed
            cached[name] = cached[name][cached[name].symbol.isin(failed)]
        table = pd.concat([cached[name], *new[name]], ignore_index=True)
        if table.empty:
            continue
        table = table.drop_duplicates(["symbol", "time"], keep="last")
        table = table[table["time"] >= oldest].sort_values(["symbol", "time"])
        table.to_parquet(LIVE_DERIVATIVES_DIR / f"{name}.parquet", index=False)
    return failed


# --- panel ---------------------------------------------------------------


def latest_decision_time(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The last training-grid decision time (hh:12, :27, :42, :57) ≤ now."""
    now = (now or _now_minute()).floor("min")
    minutes = (now.minute - GRID_OFFSET_MINUTES) % 15
    return now - pd.Timedelta(minutes=minutes)


def panel_from_bars(
    load_bars: Callable[[str], pd.DataFrame | None],
    symbols: list[str],
    decision_time: pd.Timestamp,
    derivatives_dir: Path,
    window_days: int = WINDOW_DAYS,
) -> dict:
    """The coin panel for the window ending at ``decision_time``.

    ``load_bars(symbol)`` returns a coin's 1-minute bars with the parquet's
    columns, or None (that coin's panel stays NaN). Coins are loaded one at a
    time and only BTC's bars are kept (every coin's features pair it with
    BTC), so memory stays small. Rows are 15 minutes apart and end exactly at
    ``decision_time``.
    """
    end = decision_time
    start = end - pd.Timedelta(days=window_days)
    minute_grid = pd.date_range(start, end, freq="1min", tz="UTC")
    minute_grid.name = "open_time_dt"

    def in_window(bars):
        return bars[(bars.index >= start) & (bars.index <= end)]

    btc = load_bars("BTCUSDT")
    if btc is None:
        raise RuntimeError("no BTCUSDT bars; every coin's features need them")
    btc = in_window(btc)

    def outputs():
        for symbol in symbols:
            bars = btc if symbol == "BTCUSDT" else load_bars(symbol)
            if bars is None:
                yield symbol, None
                continue
            bars = in_window(bars)
            yield (
                symbol,
                coin_panel.coin_features(
                    symbol,
                    minute_grid,
                    read=lambda s, bars=bars: btc if s == "BTCUSDT" else bars,
                ),
            )

    panel = coin_panel.assemble_panel(symbols, minute_grid, outputs())
    panel["derivatives"] = coin_panel.derivatives_features(panel, derivatives_dir)
    return panel


def live_panel(
    client: Client,
    decision_time: pd.Timestamp | None = None,
    symbols: list[str] | None = None,
) -> dict:
    """Refresh the caches and build the panel ending at the latest decision.

    An empty cache (a cold start) is filled from the APIs, which takes about
    45 minutes.
    """
    decision_time = decision_time or latest_decision_time()
    symbols = symbols or dataset_symbols()
    trading = trading_spot_symbols(client, symbols)
    until = decision_time + pd.Timedelta(minutes=1)  # bars up to the decision
    last_bar = {}
    for symbol in trading:
        bars = update_klines(client, symbol, until)
        if len(bars):
            last_bar[symbol] = bars.index[-1]
    failed = update_derivatives(client, perpetual_of(client, trading), until)

    def load_bars(symbol):
        if symbol not in last_bar:
            return None
        return with_htf(pd.read_parquet(KLINES_DIR / f"{symbol}.parquet"))

    panel = panel_from_bars(load_bars, symbols, decision_time, LIVE_DERIVATIVES_DIR)
    stale = [
        s
        for s, last in last_bar.items()
        if last < decision_time - pd.Timedelta(minutes=5)
    ]
    panel["stale_symbols"] = sorted(set(stale) | set(failed))
    return panel


# --- parity check --------------------------------------------------------


def parity(day: str, window_days: int = WINDOW_DAYS) -> None:
    """Live code path on archive data vs the full panel at the same times."""
    decision_time = latest_decision_time(pd.Timestamp(day, tz="UTC"))
    symbols = dataset_symbols()

    def load_bars(symbol):
        bars = coin_panel.read_symbol(symbol, include_holdout=True)
        start = decision_time - pd.Timedelta(days=CACHE_DAYS)
        bars = bars[(bars.index >= start) & (bars.index <= decision_time)]
        # The HTF columns recomputed over the cached range, as live
        return with_htf(bars) if len(bars) else None

    window = panel_from_bars(
        load_bars, symbols, decision_time, coin_panel.DERIVATIVES_DIR, window_days
    )
    full = coin_panel.load_panel(include_holdout=True)
    rows = np.searchsorted(full["decision_times"], window["decision_times"])
    assert np.array_equal(full["decision_times"][rows], window["decision_times"])
    full_derivatives = coin_panel.derivatives_features(
        {"symbols": full["symbols"], "decision_times": full["decision_times"][rows]}
    )[0]
    live_x, names = coin_panel.model_features(window, window["derivatives"])
    full_panel = {
        k: (
            v[rows]
            if k
            in ("close", "features", "lag_returns", "norm_vol", "tradable", "macro")
            else v
        )
        for k, v in full.items()
    }
    full_x, _ = coin_panel.model_features(
        full_panel, (full_derivatives, coin_panel.DERIVATIVE_NAMES)
    )
    # Compare over the last 14 days (the part the trader uses)
    recent = window["decision_times"] >= (
        window["decision_times"][-1] - np.timedelta64(14, "D")
    )
    usable = window["tradable"][recent] & full_panel["tradable"][recent]
    print(
        f"Window ending {decision_time}: {usable.any(0).sum()} coins, "
        f"{recent.sum()} rows compared (last 14 days)"
    )
    print(f"{'feature':<32} {'corr':>8} {'median |diff|/sd':>18} {'p99':>8}")
    for k, name in enumerate(names):
        a = live_x[recent][:, :, k][usable]
        b = full_x[recent][:, :, k][usable]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            print(f"{name:<32} {'n/a':>8}")
            continue
        sd = np.std(b[ok]) + 1e-12
        diff = np.abs(a[ok] - b[ok]) / sd
        corr = np.corrcoef(a[ok], b[ok])[0, 1] if np.std(a[ok]) > 0 else np.nan
        print(
            f"{name:<32} {corr:8.5f} {np.median(diff):18.2e} {np.percentile(diff, 99):8.2e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--update", action="store_true", help="Refresh the caches")
    parser.add_argument("--parity", default=None, help="Day (UTC) for the parity check")
    args = parser.parse_args()
    if args.parity:
        parity(args.parity)
    elif args.update:
        panel = live_panel(Client())
        print(
            f"Panel ending {panel['decision_times'][-1]}: "
            f"{panel['tradable'][-1].sum()} tradable coins; stale: {panel['stale_symbols']}"
        )


if __name__ == "__main__":
    main()
