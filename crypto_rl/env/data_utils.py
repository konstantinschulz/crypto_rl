import gc
import json
import os
import time
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.env.feature_utils import precalculate_static_obs


class CachePaths:
    """Holds paths to memory-mapped files for a given cache prefix."""

    is_built: bool = False
    lock_dir: Path
    p_prices: Path
    p_static: Path
    p_norm: Path
    p_names: Path

    @classmethod
    def get_paths(cls) -> Generator[Path, None, None]:
        """Return the paths to the memory-mapped files."""
        if not cls.is_built:
            raise RuntimeError(
                "CachePaths is not built yet. Call init_cache_paths() first."
            )
        yield from [cls.p_prices, cls.p_static, cls.p_norm, cls.p_names]


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


def init_cache_paths(config: RLConfig, cache_prefix: str = "") -> None:
    """Initialize CachePaths for a given prefix and config."""
    cache_dir = Path(config.data_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    CachePaths.lock_dir = cache_dir / f"{cache_prefix}.lock"
    CachePaths.p_prices = cache_dir / f"{cache_prefix}_prices.npy"
    CachePaths.p_static = cache_dir / f"{cache_prefix}_static.npy"
    CachePaths.p_norm = cache_dir / f"{cache_prefix}_norm_vol.npy"
    CachePaths.p_names = cache_dir / f"{cache_prefix}_names.json"
    CachePaths.is_built = True


def is_cache_built(config: RLConfig, cache_prefix: str) -> bool:
    """Check if all memory-mapped files exist for a given prefix."""
    if not cache_prefix:
        return False
    if not CachePaths.is_built:
        init_cache_paths(config, cache_prefix)
    return all(path.exists() for path in CachePaths.get_paths())


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
    if "htf_slope_15m" in long_df.columns:
        htf_slope_15m_df = _pivot_and_align("htf_slope_15m", long_df)
    else:
        ema_15 = close_df.ewm(span=15, adjust=False).mean()
        htf_slope_15m_df = (
            ((ema_15 - ema_15.shift(15)) / (close_df + 1e-8))
            .fillna(0.0)
            .astype(np.float32)
        )

    if "htf_slope_1h" in long_df.columns:
        htf_slope_1h_df = _pivot_and_align("htf_slope_1h", long_df)
    else:
        ema_60 = close_df.ewm(span=60, adjust=False).mean()
        htf_slope_1h_df = (
            ((ema_60 - ema_60.shift(60)) / (close_df + 1e-8))
            .fillna(0.0)
            .astype(np.float32)
        )

    if "htf_regime_24h" in long_df.columns:
        htf_regime_24h_df = _pivot_and_align("htf_regime_24h", long_df)
    else:
        ema_1440 = close_df.ewm(span=1440, adjust=False).mean()
        htf_regime_24h_df = (
            ((close_df - ema_1440) / (ema_1440 + 1e-8)).fillna(0.0).astype(np.float32)
        )

    # Destroy the remaining skeleton of the source DataFrame (symbol, open_time)
    del long_df
    gc.collect()

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
        if not CachePaths.is_built:
            init_cache_paths(config, cache_prefix)

        def _load_cache():
            with open(CachePaths.p_names, "r") as f:
                asset_names = json.load(f)
            prices_arr = np.load(CachePaths.p_prices, mmap_mode="r")
            static_obs = np.load(CachePaths.p_static, mmap_mode="r")
            norm_vol_arr = np.load(CachePaths.p_norm, mmap_mode="r")
            return prices_arr, static_obs, norm_vol_arr, asset_names

        # If cache is already fully built, load it and exit
        if is_cache_built(config, cache_prefix):
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
                os.mkdir(CachePaths.lock_dir)
                break  # We acquired the lock! Proceed to compute.
            except FileExistsError:
                # Another worker is currently computing this cache.
                print(
                    f"[PID {os.getpid()}] Waiting for another worker to build cache '{cache_prefix}'..."
                )
                time.sleep(10)
                # Check if the other worker finished while we were sleeping
                if is_cache_built(config, cache_prefix):
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
                (CachePaths.p_prices, prices_arr),
                (CachePaths.p_static, static_obs),
                (CachePaths.p_norm, norm_vol_arr),
            ]
            # Save atomically to prevent partial reads
            for p, arr in paths_with_arrays:
                # Ensure the temp file ends in .npy so np.save() doesn't mutate the name
                tmp_path = p.with_name(p.stem + "_tmp.npy")
                np.save(tmp_path, arr)
                tmp_path.rename(p)
            tmp_json = CachePaths.p_names.with_suffix(".tmp")
            with open(tmp_json, "w") as f:
                json.dump(asset_names, f)
            tmp_json.rename(CachePaths.p_names)
            print(
                f"[PID {os.getpid()}] Memory map cache for '{cache_prefix}' built successfully."
            )
            del temp_env, prices_arr, static_obs, norm_vol_arr
            gc.collect()
        finally:
            # ALWAYS release the lock, even if computation fails
            try:
                os.rmdir(CachePaths.lock_dir)
            except OSError:
                pass
        # Reload instantly via OS Page Cache
        return _load_cache()
    # Fallback if no cache_prefix is provided
    temp_env, norm_vol_arr, asset_names = create_temporary_environment_from_long_df(
        long_df, config
    )
    return temp_env.prices_arr, temp_env.precalc_static_obs, norm_vol_arr, asset_names
