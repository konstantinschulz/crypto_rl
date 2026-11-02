"""Kraken trading costs for the dataset's coins: listings, spreads, fee tiers.

The bot trades Kraken spot against EUR (2026-10-27: MEXC and Binance no
longer serve EU consumers). The signals still come from the Binance data, so
backtests need three things from Kraken:

- which dataset coins Kraken lists against EUR (the tradable universe)
- each pair's typical bid/ask spread: a market order pays half of it
- the fee schedule, which depends on the 30-day trading volume in USD

Kraken's public API has no spread history beyond the last few minutes
(/0/public/Spread returns ~250 quote updates). Each run of this script takes
one such snapshot per pair, time-weights the relative spread over it, and
appends the result to data/kraken/spreads.csv. Spreads vary by time of day,
so run it a few times over a day or two (e.g. from cron); `load_costs` uses
the per-pair median over all snapshots.

Usage:
    .conda/bin/python scripts/kraken_costs.py           # one snapshot
    .conda/bin/python scripts/kraken_costs.py --summary # print the table
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

repo_root = Path(__file__).resolve().parent.parent
KRAKEN_DIR = repo_root / "data" / "kraken"
PAIRS_PATH = KRAKEN_DIR / "pairs.csv"
SPREADS_PATH = KRAKEN_DIR / "spreads.csv"
API_URL = "https://api.kraken.com/0/public"
QUOTE = "ZEUR"
# Kraken's legacy asset codes for the coins whose names differ from Binance's
KRAKEN_ASSET_ALIASES = {
    "XXBT": "BTC",
    "XBT": "BTC",
    "XXDG": "DOGE",
    "XDG": "DOGE",
    "XETH": "ETH",
    "XXRP": "XRP",
    "XLTC": "LTC",
    "XXLM": "XLM",
    "XXMR": "XMR",
    "XETC": "ETC",
    "XXTZ": "XTZ",
}

# Spot fee schedule (https://www.kraken.com/features/fee-schedule, checked
# 2026-10-27): (30-day volume in USD from, maker, taker). The API's
# AssetPairs fee fields are empty for EU accounts, hence the table.
FEE_TIERS_USD = [
    (0, 0.0040, 0.0080),
    (2_500, 0.0030, 0.0060),
    (10_000, 0.0022, 0.0038),
    (25_000, 0.0020, 0.0035),
    (50_000, 0.0015, 0.0030),
    (100_000, 0.0012, 0.0025),
    (250_000, 0.0010, 0.0022),
    (500_000, 0.0008, 0.0020),
    (1_000_000, 0.0006, 0.0018),
]


def fee_rates(monthly_volume_usd: float) -> tuple[float, float]:
    """(maker, taker) fee for a 30-day volume."""
    maker, taker = FEE_TIERS_USD[0][1:]
    for threshold, tier_maker, tier_taker in FEE_TIERS_USD:
        if monthly_volume_usd >= threshold:
            maker, taker = tier_maker, tier_taker
    return maker, taker


def _get(session: requests.Session, endpoint: str, **params) -> dict:
    for attempt in range(5):
        response = session.get(f"{API_URL}/{endpoint}", params=params, timeout=30)
        payload = response.json()
        if not payload["error"]:
            return payload["result"]
        # Public endpoints allow about one call per second
        if any("Too many requests" in e for e in payload["error"]):
            time.sleep(2 + attempt * 2)
            continue
        raise RuntimeError(f"{endpoint} {params}: {payload['error']}")
    raise RuntimeError(f"{endpoint} {params}: rate limited")


def euro_pairs(session: requests.Session, spot_symbols: list[str]) -> pd.DataFrame:
    """The online EUR pair of each dataset coin that Kraken lists."""
    rows = []
    for pair, info in _get(session, "AssetPairs").items():
        if info["quote"] != QUOTE or info.get("status") != "online":
            continue
        base = KRAKEN_ASSET_ALIASES.get(info["base"], info["base"])
        symbol = f"{base}USDT"
        if symbol in spot_symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "pair": pair,
                    "costmin_eur": float(info["costmin"]),
                    "ordermin": float(info["ordermin"]),
                    "lot_decimals": int(info["lot_decimals"]),
                }
            )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def spread_snapshot(session: requests.Session, pair: str) -> dict:
    """Time-weighted relative spread over the last ~250 quote updates."""
    result = _get(session, "Spread", pair=pair)
    quotes = np.array(next(v for k, v in result.items() if k != "last"), dtype=float)
    times, bids, asks = quotes[:, 0], quotes[:, 1], quotes[:, 2]
    relative = (asks - bids) / ((asks + bids) / 2)
    # Each quote holds until the next one; the last one until now
    durations = np.diff(np.append(times, max(time.time(), times[-1] + 1)))
    durations = np.maximum(durations, 1e-3)
    return {
        "spread_bp": 1e4 * float(np.average(relative, weights=durations)),
        "window_minutes": float((time.time() - times[0]) / 60),
    }


def load_costs() -> pd.DataFrame:
    """Per Kraken-listed dataset coin: pair, median spread (bp), snapshots."""
    pairs = pd.read_csv(PAIRS_PATH)
    spreads = pd.read_csv(SPREADS_PATH)
    per_pair = spreads.groupby("symbol")["spread_bp"].agg(["median", "count"])
    per_pair.columns = ["spread_bp", "snapshots"]
    return pairs.join(per_pair, on="symbol")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--summary", action="store_true", help="Print the cost table and exit"
    )
    args = parser.parse_args()

    if args.summary:
        costs = load_costs()
        print(costs.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print(
            f"\n{len(costs)} coins; spread median {costs.spread_bp.median():.1f} bp, "
            f"p75 {costs.spread_bp.quantile(0.75):.1f} bp"
        )
        return

    # The model's coins (the coin panel, or the paper model on a VM)
    from live_data import dataset_symbols

    spot_symbols = sorted(dataset_symbols())
    session = requests.Session()
    pairs = euro_pairs(session, spot_symbols)
    KRAKEN_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(PAIRS_PATH, index=False)
    missing = sorted(set(spot_symbols) - set(pairs.symbol))
    print(
        f"{len(pairs)} of {len(spot_symbols)} coins on Kraken in EUR; missing: {missing}"
    )

    taken_at = pd.Timestamp.now(tz="UTC").floor("s")
    rows = []
    for symbol, pair in zip(pairs.symbol, pairs.pair):
        rows.append(
            {"taken_at": taken_at, "symbol": symbol, **spread_snapshot(session, pair)}
        )
        time.sleep(1.0)
    snapshot = pd.DataFrame(rows)
    snapshot.to_csv(
        SPREADS_PATH, mode="a", header=not SPREADS_PATH.exists(), index=False
    )
    print(
        f"Snapshot {taken_at}: spread median {snapshot.spread_bp.median():.1f} bp, "
        f"p75 {snapshot.spread_bp.quantile(0.75):.1f} bp, "
        f"max {snapshot.spread_bp.max():.1f} bp ({snapshot.loc[snapshot.spread_bp.idxmax(), 'symbol']})"
    )


if __name__ == "__main__":
    sys.exit(main())
