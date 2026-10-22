# -*- coding: utf-8 -*-
"""Data loading utilities for experiment."""

import ctypes
import fcntl
import gc
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import optuna
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.data import FoldSplit, make_walk_forward_splits, read_n_rows
from crypto_rl.env.data_utils import parquet_open_time_range, split_cache_name
from crypto_rl.env.logging_utils import LoggerBase


def load_raw_data(
    config: RLConfig, logger: LoggerBase, trial: optuna.trial.Trial | None
) -> pd.DataFrame:
    """Load raw OHLCV data and configure evaluation frequency.
    Returns the raw DataFrame.
    """
    logger.print_if_not_trial(logging.DEBUG, trial, "1. Loading raw data...")
    raw_df = read_n_rows(str(config.parquet_path), config.n_rows)
    logger.print_if_not_trial(
        logging.DEBUG,
        trial,
        f"Evaluation frequency: after every {config.eval_freq} steps",
    )
    return raw_df


@contextmanager
def raw_data_lock(config: RLConfig) -> Iterator[None]:
    """Hold an exclusive lock while this process loads the raw data.

    Loading 18M rows peaks at about 5 GB RSS per process, and glibc keeps
    about 1.6 GB of it afterwards. Several Optuna workers doing it at once
    run the machine out of memory, so they take turns. The kernel releases
    the lock when the process exits, even on SIGKILL. After acquiring it,
    callers should re-check their cache: another worker may have built it
    while this one waited.
    """
    lock_path = Path(config.data_cache_dir) / "raw_data.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def release_freed_memory() -> None:
    """Collect garbage and ask glibc to return freed heap memory to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def splits_cache_path(config: RLConfig) -> Path:
    """Path of the walk-forward split cache for this config and data file."""
    data_range = parquet_open_time_range(config.parquet_path)
    return Path(config.data_cache_dir) / split_cache_name(config, data_range)


def read_cached_splits(config: RLConfig) -> list[FoldSplit] | None:
    """The cached walk-forward splits, or None if they haven't been built."""
    path = splits_cache_path(config)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return [FoldSplit(*split) for split in json.load(f)]


def write_cached_splits(config: RLConfig, splits: list[FoldSplit]) -> None:
    """Save the walk-forward splits atomically for later runs and other workers."""
    path = splits_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w") as f:
        json.dump([list(split) for split in splits], f)
    tmp_path.rename(path)


def prepare_splits(
    raw_df: pd.DataFrame,
    config: RLConfig,
    logger: LoggerBase,
    trial: optuna.trial.Trial | None,
) -> list[FoldSplit]:
    """Create walk‑forward train / val / test splits based on config."""
    splits = make_walk_forward_splits(
        raw_df["open_time"].unique(),
        n_folds=config.cv_folds,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
    )
    logger.print_if_not_trial(
        logging.DEBUG,
        trial,
        f"Dataset split into {len(splits)}-fold walk-forward cross-validation "
        f"(val_fraction={config.val_fraction}, test_fraction={config.test_fraction}).",
    )
    return splits
