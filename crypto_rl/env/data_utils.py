import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.data import FoldSplit, open_time_to_datetime
from crypto_rl.env.feature_utils import precalculate_static_obs

# Bump whenever the static features or the pivot logic change, so that stale
# memmaps and split files are never picked up under the same name.
FEATURE_VERSION = 2


@dataclass(frozen=True)
class CachePaths:
    """Paths to the memory-mapped files for one cache prefix."""

    cache_dir: Path
    prefix: str

    @classmethod
    def for_prefix(cls, config: RLConfig, prefix: str) -> "CachePaths":
        """Return the cache paths for `prefix` under `config.data_cache_dir`."""
        return cls(cache_dir=Path(config.data_cache_dir), prefix=prefix)

    @property
    def lock_dir(self) -> Path:
        return self.cache_dir / f"{self.prefix}.lock"

    @property
    def p_prices(self) -> Path:
        return self.cache_dir / f"{self.prefix}_prices.npy"

    @property
    def p_static(self) -> Path:
        return self.cache_dir / f"{self.prefix}_static.npy"

    @property
    def p_norm(self) -> Path:
        return self.cache_dir / f"{self.prefix}_norm_vol.npy"

    @property
    def p_names(self) -> Path:
        return self.cache_dir / f"{self.prefix}_names.json"

    def exists(self) -> bool:
        """True if all memory-mapped files for this prefix exist."""
        return all(
            p.exists() for p in (self.p_prices, self.p_static, self.p_norm, self.p_names)
        )


def format_date_range(start: int, end: int) -> str:
    """Filename-safe label for a [start, end] range of `open_time` values."""
    fmt = "%Y%m%dT%H%M%S"
    start_s = open_time_to_datetime(start).strftime(fmt)
    end_s = open_time_to_datetime(end).strftime(fmt)
    return f"{start_s}-{end_s}"


def parquet_open_time_range(path: str) -> tuple[int, int]:
    """Min and max `open_time` in a parquet file, read from row-group statistics.

    Falls back to scanning the column when the statistics are missing.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.get_field_index("open_time")
    lo, hi = None, None
    for rg in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg).column(col).statistics
        if stats is None or not stats.has_min_max:
            lo, hi = None, None
            break
        lo = stats.min if lo is None else min(lo, stats.min)
        hi = stats.max if hi is None else max(hi, stats.max)
    if lo is None or hi is None:
        table = ds.dataset(path, format="parquet").to_table(columns=["open_time"])
        mm = pc.min_max(table.column("open_time")).as_py()
        lo, hi = mm["min"], mm["max"]
    return int(lo), int(hi)


def split_cache_name(config: RLConfig, data_range: tuple[int, int]) -> str:
    """File name of the walk-forward split cache for this config and data range."""
    return (
        f"seed_{config.data_seed}_rows_{config.n_rows}_folds_{config.cv_folds}"
        f"_val_{config.val_fraction}_test_{config.test_fraction}"
        f"_feat_v{FEATURE_VERSION}_data_{format_date_range(*data_range)}_splits.json"
    )


class FoldCachePrefixes(NamedTuple):
    """Memmap prefixes of one fold's three slices."""

    train: str
    val: str
    test: str


def fold_cache_prefixes(
    config: RLConfig,
    num_assets: int,
    fold_idx: int,
    data_range: tuple[int, int],
    split: FoldSplit,
) -> FoldCachePrefixes:
    """Memmap prefixes (train, val, test) for one walk-forward fold.

    The train slice spans the start of the data up to `split.t_train_max`; the
    val and test slices span their own `[min, max]` ranges. The static features
    depend on `window_size` (rolling volatility, momentum, z-scores, VWAP), so
    it is part of the name.
    """
    base = (
        f"seed_{config.data_seed}_rows_{config.n_rows}_assets_{num_assets}"
        f"_folds_{config.cv_folds}_val_{config.val_fraction}"
        f"_test_{config.test_fraction}_win_{config.window_size}"
        f"_feat_v{FEATURE_VERSION}_fold_{fold_idx}"
    )
    return FoldCachePrefixes(
        train=f"{base}_train_{format_date_range(data_range[0], split.t_train_max)}",
        val=f"{base}_val_{format_date_range(split.t_val_min, split.t_val_max)}",
        test=f"{base}_test_{format_date_range(split.t_test_min, split.t_test_max)}",
    )


def fold_slice_bounds(split: FoldSplit) -> dict[str, tuple[int | None, int]]:
    """Inclusive `open_time` bounds per slice; `None` means unbounded below."""
    return {
        "train": (None, split.t_train_max),
        "val": (split.t_val_min, split.t_val_max),
        "test": (split.t_test_min, split.t_test_max),
    }


def assert_chronological_bounds(bounds: dict[str, tuple[int, int]]) -> None:
    """Raise unless the `(first, last)` open_time ranges are non-empty, disjoint
    and in the given order (oldest first, e.g. train, val, test).
    """
    names = list(bounds)
    for earlier, later in zip(names, names[1:]):
        if bounds[earlier][1] >= bounds[later][0]:
            raise AssertionError(
                f"Slices '{earlier}' {bounds[earlier]} and '{later}' "
                f"{bounds[later]} overlap in time; the walk-forward split is broken."
            )
    firsts = [first for first, _ in bounds.values()]
    if len(set(firsts)) != len(firsts):
        raise AssertionError(
            f"Slices {names} do not all start at distinct open_time values: {firsts}"
        )


def build_fold_caches(
    raw_df: pd.DataFrame,
    config: RLConfig,
    split: FoldSplit,
    prefixes: FoldCachePrefixes,
    names: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Build the missing memmap caches of one fold, one slice at a time.

    First checks that the requested slices are disjoint and chronological, so
    a broken split can never silently train and evaluate on the same data.
    """
    bounds = fold_slice_bounds(split)
    open_time = raw_df["open_time"]
    observed: dict[str, tuple[int, int]] = {}
    for name in names:
        lo, hi = bounds[name]
        mask = open_time <= hi if lo is None else (open_time >= lo) & (open_time <= hi)
        selected = open_time[mask]
        if selected.empty:
            raise AssertionError(f"Fold slice '{name}' is empty.")
        observed[name] = (int(selected.min()), int(selected.max()))
    assert_chronological_bounds(observed)

    for name in names:
        prefix = getattr(prefixes, name)
        if is_cache_built(config, prefix):
            continue
        lo, hi = bounds[name]
        mask = open_time <= hi if lo is None else (open_time >= lo) & (open_time <= hi)
        slice_df = raw_df[mask].copy().reset_index(drop=True)
        compute_static_obs_from_long_df(slice_df, config, cache_prefix=prefix)
        del slice_df
        gc.collect()


def assert_distinct_first_timestamps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise if the two slices both start at the same `open_time`."""
    train_first = train_df["open_time"].min()
    test_first = test_df["open_time"].min()
    if train_first == test_first:
        raise AssertionError(
            f"Both slices start at open_time={train_first}; "
            "the walk-forward split is broken."
        )


def create_temporary_environment_from_long_df(
    long_df, config
) -> tuple[SimpleNamespace, np.ndarray, list[str]]:
    """Pivot long-format OHLCV DataFrame and pre-calculate static observations.
    Returns a temporary SimpleNamespace with the pivoted DataFrames and pre-calculated static observations."""
    (
        close_df,
        open_df,
        high_df,
        low_df,
        volume_df,
        htf_slope_15m_df,
        htf_slope_1h_df,
        htf_regime_24h_df,
    ) = pivot_ohlcv(long_df)
    asset_names = close_df.columns.tolist()
    num_assets = len(asset_names)
    tr = high_df - low_df
    atr = tr.rolling(window=14, min_periods=1).mean()
    norm_vol_arr = (
        (atr / close_df)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1e-8)
        .values.astype(np.float32)
    )
    temp_env = SimpleNamespace(
        prices_df=close_df,
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        htf_slope_15m_df=htf_slope_15m_df,
        htf_slope_1h_df=htf_slope_1h_df,
        htf_regime_24h_df=htf_regime_24h_df,
        config=config,
        num_assets=num_assets,
    )
    precalculate_static_obs(temp_env)
    # After this call, temp_env.prices_df / open_df / … are all None (freed
    # inside precalculate_static_obs).  Only numpy arrays remain.
    return temp_env, norm_vol_arr, asset_names


def is_cache_built(config: RLConfig, cache_prefix: str) -> bool:
    """Check if all memory-mapped files exist for a given prefix."""
    if not cache_prefix:
        return False
    return CachePaths.for_prefix(config, cache_prefix).exists()


def pivot_ohlcv(long_df: pd.DataFrame):
    """
    Pivot OHLCV and HTF columns into separate DataFrames for each asset.
    Destroys the source DataFrame incrementally to prevent massive RAM spikes.
    """

    def _pivot_and_align(val_col: str, df: pd.DataFrame) -> pd.DataFrame:
        pivoted = df.pivot(index="open_time", columns="symbol", values=val_col)

        # Destroy the column in the original long_df immediately to prevent memory doubling
        if val_col in df.columns:
            del df[val_col]
            gc.collect()

        # Forward-fill / back-fill missing values, then cast to float32
        return pivoted.ffill().bfill().fillna(0.0).astype(np.float32)

    close_df = _pivot_and_align("close", long_df)
    open_df = _pivot_and_align("open", long_df)
    high_df = _pivot_and_align("high", long_df)
    low_df = _pivot_and_align("low", long_df)
    volume_df = _pivot_and_align("volume", long_df)

    # Handle HTF indicator columns if present in long_df, otherwise fallback compute
    htf_slope_15m_df = (
        _pivot_and_align("htf_slope_15m", long_df)
        if "htf_slope_15m" in long_df.columns
        else None
    )
    htf_slope_1h_df = (
        _pivot_and_align("htf_slope_1h", long_df)
        if "htf_slope_1h" in long_df.columns
        else None
    )
    htf_regime_24h_df = (
        _pivot_and_align("htf_regime_24h", long_df)
        if "htf_regime_24h" in long_df.columns
        else None
    )

    # Destroy the remaining skeleton of the source DataFrame (symbol, open_time)
    del long_df
    gc.collect()

    return _with_htf_fallbacks(
        close_df,
        open_df,
        high_df,
        low_df,
        volume_df,
        htf_slope_15m_df,
        htf_slope_1h_df,
        htf_regime_24h_df,
    )


def pivot_ohlcv_wide(wide: dict[str, pd.DataFrame]):
    """Finalise per-column wide frames produced by ``load_window_wide``.

    Applies the same forward/back-fill, float32 cast and HTF fallbacks as
    :func:`pivot_ohlcv`, without ever building a long-format frame.
    """

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        return df.ffill().bfill().fillna(0.0).astype(np.float32)

    def _align_optional(col: str) -> pd.DataFrame | None:
        return _align(wide[col]) if col in wide else None

    return _with_htf_fallbacks(
        _align(wide["close"]),
        _align(wide["open"]),
        _align(wide["high"]),
        _align(wide["low"]),
        _align(wide["volume"]),
        _align_optional("htf_slope_15m"),
        _align_optional("htf_slope_1h"),
        _align_optional("htf_regime_24h"),
    )


def _with_htf_fallbacks(
    close_df,
    open_df,
    high_df,
    low_df,
    volume_df,
    htf_slope_15m_df,
    htf_slope_1h_df,
    htf_regime_24h_df,
):
    """Compute HTF indicators from close when they were not supplied."""
    if htf_slope_15m_df is None:
        ema_15 = close_df.ewm(span=15, adjust=False).mean()
        htf_slope_15m_df = (
            ((ema_15 - ema_15.shift(15)) / (close_df + 1e-8))
            .fillna(0.0)
            .astype(np.float32)
        )
    if htf_slope_1h_df is None:
        ema_60 = close_df.ewm(span=60, adjust=False).mean()
        htf_slope_1h_df = (
            ((ema_60 - ema_60.shift(60)) / (close_df + 1e-8))
            .fillna(0.0)
            .astype(np.float32)
        )
    if htf_regime_24h_df is None:
        ema_1440 = close_df.ewm(span=1440, adjust=False).mean()
        htf_regime_24h_df = (
            ((close_df - ema_1440) / (ema_1440 + 1e-8)).fillna(0.0).astype(np.float32)
        )

    return (
        close_df,
        open_df,
        high_df,
        low_df,
        volume_df,
        htf_slope_15m_df,
        htf_slope_1h_df,
        htf_regime_24h_df,
    )


def compute_static_obs_from_long_df(
    long_df: pd.DataFrame | None, config: RLConfig, cache_prefix: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Pivot long-format OHLCV DataFrame and pre-calculate static observations.

    Returns:
        tuple: (prices_arr, static_obs, norm_vol_arr, asset_names)

    Memory notes
    ------------
    - The pivoted DataFrames are float32 (not float64) thanks to pivot_ohlcv.
    - precalculate_static_obs converts them to numpy *and* sets the DataFrame
      slots to None, so they are freed before this function returns.
    """
    if cache_prefix:
        paths = CachePaths.for_prefix(config, cache_prefix)
        paths.cache_dir.mkdir(parents=True, exist_ok=True)

        def _load_cache():
            with open(paths.p_names, "r") as f:
                asset_names = json.load(f)
            prices_arr = np.load(paths.p_prices, mmap_mode="r")
            static_obs = np.load(paths.p_static, mmap_mode="r")
            norm_vol_arr = np.load(paths.p_norm, mmap_mode="r")
            return prices_arr, static_obs, norm_vol_arr, asset_names

        # If cache is already fully built, load it and exit
        if paths.exists():
            res = _load_cache()
            del long_df
            gc.collect()
            return res
        if long_df is None:
            raise RuntimeError(
                f"Cache '{cache_prefix}' is missing, but no raw DataFrame was provided to build it! "
                "This usually indicates a race condition or manual cache deletion mid-run."
            )
        # ATOMIC LOCKING: Prevent RAM explosion from concurrent builds
        while True:
            try:
                # os.mkdir is guaranteed atomic by the OS (POSIX and Windows)
                os.mkdir(paths.lock_dir)
                break  # We acquired the lock! Proceed to compute.
            except FileExistsError:
                # Another worker is currently computing this cache.
                print(
                    f"[PID {os.getpid()}] Waiting for another worker to build cache '{cache_prefix}'..."
                )
                time.sleep(10)
                # Check if the other worker finished while we were sleeping
                if paths.exists():
                    res = _load_cache()
                    del long_df
                    gc.collect()
                    return res
        # CRITICAL SECTION (Only one worker ever gets here)
        try:
            print(
                f"[PID {os.getpid()}] Acquired lock. Building memory map cache for '{cache_prefix}'..."
            )
            temp_env, norm_vol_arr, asset_names = (
                create_temporary_environment_from_long_df(long_df, config)
            )
            prices_arr = temp_env.prices_arr
            static_obs = temp_env.precalc_static_obs
            paths_with_arrays: list[tuple[Path, np.ndarray]] = [
                (paths.p_prices, prices_arr),
                (paths.p_static, static_obs),
                (paths.p_norm, norm_vol_arr),
            ]
            # Save atomically to prevent partial reads
            for p, arr in paths_with_arrays:
                # Ensure the temp file ends in .npy so np.save() doesn't mutate the name
                tmp_path = p.with_name(p.stem + "_tmp.npy")
                np.save(tmp_path, arr)
                tmp_path.rename(p)
            tmp_json = paths.p_names.with_suffix(".tmp")
            with open(tmp_json, "w") as f:
                json.dump(asset_names, f)
            tmp_json.rename(paths.p_names)
            print(
                f"[PID {os.getpid()}] Memory map cache for '{cache_prefix}' built successfully."
            )
            del temp_env, prices_arr, static_obs, norm_vol_arr
            gc.collect()
        finally:
            # ALWAYS release the lock, even if computation fails
            try:
                os.rmdir(paths.lock_dir)
            except OSError:
                pass
        # Reload instantly via OS Page Cache
        return _load_cache()
    # Fallback if no cache_prefix is provided
    temp_env, norm_vol_arr, asset_names = create_temporary_environment_from_long_df(
        long_df, config
    )
    return temp_env.prices_arr, temp_env.precalc_static_obs, norm_vol_arr, asset_names
