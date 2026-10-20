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
from crypto_rl.data import DEFAULT_SYMBOLS, make_walk_forward_splits, read_n_rows
from crypto_rl.env.data_utils import (
    build_fold_caches,
    fold_cache_prefixes,
    is_cache_built,
    parquet_open_time_range,
)

# Must match the values used in optuna_search.py objective()
cfg: RLConfig = get_base_config()
num_assets = len(DEFAULT_SYMBOLS)

config = RLConfig(
    n_rows=cfg.n_rows,
    data_seed=cfg.data_seed,
    cv_folds=cfg.cv_folds,
    test_fraction=cfg.test_fraction,
    val_fraction=cfg.val_fraction,
    parquet_path=cfg.parquet_path,
)
data_range = parquet_open_time_range(config.parquet_path)

print("[Cache] Loading raw data to derive split bounds...")
np.random.seed(cfg.data_seed)
raw_df = read_n_rows(str(config.parquet_path), n_rows=cfg.n_rows)

# Build walk-forward train / val / test splits
splits = make_walk_forward_splits(
    raw_df["open_time"].unique(),
    n_folds=cfg.cv_folds,
    test_fraction=config.test_fraction,
    val_fraction=config.val_fraction,
)

for fold_idx, split in enumerate(splits):
    prefixes = fold_cache_prefixes(config, num_assets, fold_idx, data_range, split)
    if all(is_cache_built(config, getattr(prefixes, n)) for n in prefixes._fields):
        print(f"[Cache] Fold {fold_idx + 1} caches already exist — skipping.")
        continue
    print(f"[Cache] Building missing fold {fold_idx + 1} caches...")
    build_fold_caches(raw_df, config, split, prefixes)

del raw_df
gc.collect()
print("[Cache] All memmap caches ready.")
