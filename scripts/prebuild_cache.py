"""Build the walk-forward splits and memmap caches for the Optuna workers.

Run once, before the workers start (scripts/run_optuna_workers.sh does). With
the split JSON and the train / val memmaps on disk, workers never load the raw
parquet, which peaks at about 5 GB RSS per process at 18M rows.
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from optuna_search import get_base_config

from crypto_rl.data import DEFAULT_SYMBOLS
from crypto_rl.env.data_utils import (
    build_fold_caches,
    fold_cache_prefixes,
    is_cache_built,
    parquet_open_time_range,
)
from crypto_rl.experiment.data_loading import (
    load_raw_data,
    prepare_splits,
    raw_data_lock,
    read_cached_splits,
    release_freed_memory,
    write_cached_splits,
)


class PrintLogger:
    def debug(self, msg):
        print(msg)

    def info(self, msg):
        print(msg)

    def log(self, msg, trial=None):
        print(msg)

    def print_if_not_trial(self, level, trial, msg):
        print(msg)


# The workers' exact config, so the split and cache names match theirs
config = get_base_config()
# Optuna trials never read the test slices
CACHE_SLICES = ("train", "val")
logger = PrintLogger()
data_range = parquet_open_time_range(config.parquet_path)


def fold_prefixes(splits):
    return [
        fold_cache_prefixes(config, len(DEFAULT_SYMBOLS), fold_idx, data_range, split)
        for fold_idx, split in enumerate(splits)
    ]


def missing(splits) -> bool:
    return splits is None or any(
        not is_cache_built(config, getattr(prefixes, name))
        for prefixes in fold_prefixes(splits)
        for name in CACHE_SLICES
    )


with raw_data_lock(config):
    splits = read_cached_splits(config)
    if not missing(splits):
        print("[Cache] Splits and caches already exist — skipping.")
        sys.exit(0)
    print(f"[PID {os.getpid()}] [Cache] Loading raw data...")
    raw_df = load_raw_data(config, logger, None)
    if splits is None:
        splits = prepare_splits(raw_df, config, logger, None)
        write_cached_splits(config, splits)
    for fold_idx, (split, prefixes) in enumerate(zip(splits, fold_prefixes(splits))):
        print(f"[Cache] Building missing fold {fold_idx + 1} caches...")
        build_fold_caches(raw_df, config, split, prefixes, CACHE_SLICES)
    del raw_df
    release_freed_memory()
print("[Cache] Splits and memmap caches ready.")
