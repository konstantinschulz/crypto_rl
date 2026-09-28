import os
import sys

from optuna_search import get_base_config

repo_root = os.path.abspath(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__) if "__file__" in dir() else os.getcwd()
        )
    )
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import gc

import numpy as np

from crypto_rl.config import RLConfig
from crypto_rl.data import DEFAULT_SYMBOLS, get_walk_forward_splits, read_n_rows
from crypto_rl.env.data_utils import compute_static_obs_from_long_df, is_cache_built

# Must match the values used in optuna_search.py objective()
cfg: RLConfig = get_base_config()
num_assets = len(DEFAULT_SYMBOLS)

config = RLConfig(n_rows=cfg.n_rows, data_seed=cfg.data_seed, cv_folds=cfg.cv_folds)

# Check if all caches are already built
all_built = True
for fold_idx in range(cfg.cv_folds):
    for split in ["train", "test"]:
        prefix = f"seed_{cfg.data_seed}_rows_{cfg.n_rows}_assets_{num_assets}_fold_{fold_idx}_{split}"
        if not is_cache_built(config, prefix):
            all_built = False
            break

if all_built:
    print("[Cache] All memmap caches already exist — skipping pre-build.")
    sys.exit(0)

print("[Cache] Pre-building memmap cache (single process, avoids per-worker RAM spike)...")
print(f"[Cache] Loading {cfg.n_rows:,} rows from parquet...")
np.random.seed(cfg.data_seed)
raw_df = read_n_rows(str(config.parquet_path), n_rows=cfg.n_rows)

# Build walk-forward splits
unique_times = np.sort(raw_df["open_time"].unique())
n_test_times = round(len(unique_times) * config.test_fraction)
if cfg.cv_folds > 1:
    splits = get_walk_forward_splits(raw_df, n_folds=cfg.cv_folds)
else:
    splits = [
        (
            int(unique_times[-(n_test_times + 1)]),
            int(unique_times[-n_test_times]),
            int(unique_times[-1]),
            "",
            "",
        )
    ]

for fold_idx, (t_train_max, t_test_min, t_test_max, *_) in enumerate(splits):
    prefix_train = f"seed_{cfg.data_seed}_rows_{cfg.n_rows}_assets_{num_assets}_fold_{fold_idx}_train"
    prefix_test = f"seed_{cfg.data_seed}_rows_{cfg.n_rows}_assets_{num_assets}_fold_{fold_idx}_test"
    if not is_cache_built(config, prefix_train):
        print(f"[Cache] Building fold {fold_idx + 1} train cache...")
        train_df = (
            raw_df[raw_df["open_time"] <= t_train_max].copy().reset_index(drop=True)
        )
        compute_static_obs_from_long_df(train_df, config, cache_prefix=prefix_train)
        del train_df
        gc.collect()
    if not is_cache_built(config, prefix_test):
        print(f"[Cache] Building fold {fold_idx + 1} test cache...")
        test_df = (
            raw_df[
                (raw_df["open_time"] >= t_test_min)
                & (raw_df["open_time"] <= t_test_max)
            ]
            .copy()
            .reset_index(drop=True)
        )
        compute_static_obs_from_long_df(test_df, config, cache_prefix=prefix_test)
        del test_df
        gc.collect()

del raw_df
gc.collect()
print("[Cache] All memmap caches ready.")
