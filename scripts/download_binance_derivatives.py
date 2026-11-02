"""Download Binance USDⓈ-M futures data for the dataset's coins.

Prices on Binance and MEXC move together (hourly returns correlate at 0.99+,
checked 2026-10-27), and MEXC's API keeps no history of open interest or
long/short ratios and only ~1.5 years of funding rates. So the derivatives
signals come from Binance's public archive, data.binance.vision:

- metrics (daily files, 5-minute rows): open interest in coins and USDT,
  top-trader and all-account long/short ratios, taker buy/sell volume ratio
- fundingRate (monthly files): every funding settlement and its interval
- premiumIndexKlines, 15m (monthly files): the perpetual's premium over the
  spot index as OHLC, i.e. the basis

For every coin of the spot dataset with a USDⓈ-M perpetual (BONK, FLOKI,
PEPE and SHIB trade as 1000…USDT; NEXO has none), the script lists the
archive, downloads the files in the date range, verifies each against the
listing's MD5 (or the .CHECKSUM file), and writes one parquet per data type
to data/binance_derivatives/, keyed by the spot symbol. Raw zips are kept
under raw/, so a rerun only fetches what is missing.

The range covers the dataset including the fold-3 holdout (2025-02-18 →
2026-01-31), which the final report needs. Research code must cut at
coin_panel.HOLDOUT_START, as the coin panel does.

Usage (about an hour for all coins; ~1.5 GB of zips):
    .conda/bin/python scripts/download_binance_derivatives.py
    .conda/bin/python scripts/download_binance_derivatives.py --types funding --symbols BTCUSDT
"""

import argparse
import hashlib
import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import requests

repo_root = Path(__file__).resolve().parent.parent
SPOT_PARQUET = repo_root / "binance_spot_1m_last4y_single_htf.parquet"
OUTPUT_DIR = repo_root / "data" / "binance_derivatives"
ARCHIVE_URL = "https://data.binance.vision"
LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

# Per data type: archive prefix for a perpetual symbol, and how a file name
# encodes its date (daily or monthly)
DATA_TYPES = {
    "metrics": ("data/futures/um/daily/metrics/{perp}/", "daily"),
    "funding": ("data/futures/um/monthly/fundingRate/{perp}/", "monthly"),
    "premium_15m": (
        "data/futures/um/monthly/premiumIndexKlines/{perp}/15m/",
        "monthly",
    ),
}
# Binance kline CSV layout (older files have no header row)
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
# Perpetuals quoted per 1000 coins; the spot symbol is the key everywhere else
PERP_PREFIXES = ["", "1000", "1000000", "1M"]


def list_archive(session: requests.Session, prefix: str) -> dict[str, str]:
    """All zip keys under ``prefix`` with their ETag (S3 lists 1000 per page)."""
    keys, marker = {}, ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        response = session.get(LISTING_URL, params=params, timeout=60)
        response.raise_for_status()
        text = response.text
        for key, etag in re.findall(
            r"<Key>([^<]+\.zip)</Key>.*?<ETag>&quot;([^&]+)&quot;</ETag>",
            text,
            re.DOTALL,
        ):
            keys[key] = etag
        if "<IsTruncated>true</IsTruncated>" not in text:
            return keys
        marker = re.findall(r"<Key>([^<]+)</Key>", text)[-1]


def list_perpetuals(session: requests.Session) -> set[str]:
    """Symbols with a USDⓈ-M funding-rate archive, i.e. every perpetual."""
    perps, marker = set(), ""
    while True:
        response = session.get(
            LISTING_URL,
            params={
                "prefix": "data/futures/um/monthly/fundingRate/",
                "delimiter": "/",
                "marker": marker,
            },
            timeout=60,
        )
        response.raise_for_status()
        found = re.findall(
            r"<Prefix>data/futures/um/monthly/fundingRate/([^<]+)/</Prefix>",
            response.text,
        )
        perps.update(found)
        if "<IsTruncated>true</IsTruncated>" not in response.text:
            return perps
        marker = f"data/futures/um/monthly/fundingRate/{found[-1]}/"


def file_date(key: str, granularity: str) -> pd.Timestamp:
    """The day or month a file covers, from its name."""
    pattern = (
        r"(\d{4}-\d{2}-\d{2})\.zip$"
        if granularity == "daily"
        else r"(\d{4}-\d{2})\.zip$"
    )
    return pd.Timestamp(re.search(pattern, key).group(1))


def in_range(
    key: str, granularity: str, start: pd.Timestamp, end: pd.Timestamp
) -> bool:
    date = file_date(key, granularity)
    if granularity == "monthly":
        # Keep a month if it overlaps the range
        return date <= end and date + pd.offsets.MonthEnd(0) >= start
    return start <= date <= end


def download(session: requests.Session, key: str, etag: str, target: Path) -> str:
    """Fetch one zip unless a verified copy exists; returns a status string."""
    if target.exists() and _matches(target.read_bytes(), key, etag, session):
        return "cached"
    for attempt in range(4):
        try:
            response = session.get(f"{ARCHIVE_URL}/{key}", timeout=120)
            response.raise_for_status()
            if _matches(response.content, key, etag, session):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                return "downloaded"
        except requests.RequestException:
            if attempt == 3:
                raise
    raise RuntimeError(f"Checksum mismatch after retries: {key}")


def _matches(content: bytes, key: str, etag: str, session: requests.Session) -> bool:
    """Verify against the S3 ETag (an MD5 unless the upload was multipart),
    else against the archive's .CHECKSUM file (SHA-256)."""
    if "-" not in etag:
        return hashlib.md5(content).hexdigest() == etag
    checksum = session.get(f"{ARCHIVE_URL}/{key}.CHECKSUM", timeout=60).text.split()[0]
    return hashlib.sha256(content).hexdigest() == checksum


def read_zip_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """The CSV inside an archive zip; older files have no header row."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0])
    first_field = raw.split(b",", 1)[0].strip()
    has_header = not first_field.replace(b".", b"").replace(b"-", b"").isdigit()
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if not has_header:
        frame.columns = columns[: frame.shape[1]]
    return frame


def to_utc(values: pd.Series) -> pd.Series:
    """Epoch milliseconds or microseconds (the archive switched units) to UTC."""
    numbers = pd.to_numeric(values)
    unit = "us" if numbers.max() > 1e14 else "ms"
    return pd.to_datetime(numbers, unit=unit, utc=True)


def parse(data_type: str, spot_symbol: str, paths: list[Path]) -> pd.DataFrame:
    """One spot symbol's files of one type as a tidy frame."""
    frames = []
    for path in sorted(paths):
        if data_type == "metrics":
            f = read_zip_csv(
                path,
                [
                    "create_time",
                    "symbol",
                    "sum_open_interest",
                    "sum_open_interest_value",
                    "count_toptrader_long_short_ratio",
                    "sum_toptrader_long_short_ratio",
                    "count_long_short_ratio",
                    "sum_taker_long_short_vol_ratio",
                ],
            )
            f["time"] = pd.to_datetime(f.pop("create_time"), utc=True)
            f = f.drop(columns=["symbol"])
        elif data_type == "funding":
            f = read_zip_csv(
                path, ["calc_time", "funding_interval_hours", "last_funding_rate"]
            )
            f["time"] = to_utc(f.pop("calc_time"))
        else:
            f = read_zip_csv(path, KLINE_COLUMNS)
            f["time"] = to_utc(f.pop("open_time"))
            # The premium index has no volume; keep the OHLC of the premium
            f = f[["time", "open", "high", "low", "close"]]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "symbol", spot_symbol)
    numeric = [c for c in out.columns if c not in ("symbol", "time")]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    return out.drop_duplicates("time").sort_values("time")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2022-01-01", help="First day (UTC)")
    parser.add_argument("--end", default="2026-01-31", help="Last day (UTC)")
    parser.add_argument(
        "--types", nargs="*", default=list(DATA_TYPES), choices=list(DATA_TYPES)
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Spot symbols; default all"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.workers))
    spot_symbols = args.symbols or sorted(
        ds.dataset(SPOT_PARQUET)
        .to_table(columns=["symbol"])
        .column("symbol")
        .unique()
        .to_pylist()
    )
    perpetuals = list_perpetuals(session)
    perp_of = {}
    for spot in spot_symbols:
        match = next((p + spot for p in PERP_PREFIXES if p + spot in perpetuals), None)
        if match is None:
            print(f"{spot}: no USDⓈ-M perpetual, skipped")
        else:
            perp_of[spot] = match
    print(f"{len(perp_of)} of {len(spot_symbols)} coins have a perpetual", flush=True)

    for data_type in args.types:
        prefix_template, granularity = DATA_TYPES[data_type]
        jobs = []  # (spot symbol, key, etag, local path)
        for spot, perp in perp_of.items():
            listing = list_archive(session, prefix_template.format(perp=perp))
            for key, etag in listing.items():
                if in_range(key, granularity, start, end):
                    jobs.append((spot, key, etag, OUTPUT_DIR / "raw" / key))
        print(f"[{data_type}] {len(jobs)} files", flush=True)

        status_counts: dict[str, int] = {}
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(download, session, key, etag, path)
                for _, key, etag, path in jobs
            ]
            for done, future in enumerate(as_completed(futures), 1):
                status = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
                if done % 2000 == 0 or done == len(futures):
                    print(
                        f"[{data_type}] {done}/{len(futures)} {status_counts}",
                        flush=True,
                    )

        frames = []
        for spot in perp_of:
            paths = [path for s, _, _, path in jobs if s == spot]
            if paths:
                frames.append(parse(data_type, spot, paths))
        table = pd.concat(frames, ignore_index=True)
        # Monthly files can reach past the range; keep [start, end]
        in_window = (table["time"] >= start.tz_localize("UTC")) & (
            table["time"] < (end + pd.Timedelta(days=1)).tz_localize("UTC")
        )
        table = table[in_window]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{data_type}.parquet"
        table.to_parquet(out_path, index=False)
        coverage = table.groupby("symbol")["time"].agg(["min", "max", "count"])
        print(
            f"[{data_type}] saved {out_path}: {len(table)} rows, {table.symbol.nunique()} coins; "
            f"first timestamps from {coverage['min'].min():%Y-%m-%d} to "
            f"{coverage['min'].max():%Y-%m-%d}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
