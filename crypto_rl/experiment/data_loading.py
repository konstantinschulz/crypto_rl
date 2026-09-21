# -*- coding: utf-8 -*-
"""Data loading utilities for experiment.
"""

import logging
from typing import Any

import optuna
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.data import get_walk_forward_splits, read_n_rows
from crypto_rl.env.logging_utils import print_if_not_trial


def load_raw_data(config: RLConfig, logger: Any, trial: optuna.trial.Trial | None) -> pd.DataFrame:
    """Load raw OHLCV data and configure evaluation frequency.
    Returns the raw DataFrame.
    """
    print_if_not_trial(logger, logging.DEBUG, trial, "1. Loading raw data...")
    raw_df = read_n_rows(str(config.parquet_path), config.n_rows)
    if config.eval_freq == "auto":
        config.eval_freq = max(2000, config.timesteps // 10)
    print_if_not_trial(
        logger,
        logging.DEBUG,
        trial,
        f"Evaluation frequency: after every {config.eval_freq} steps",
    )
    return raw_df


def prepare_splits(
    raw_df: pd.DataFrame,
    config: RLConfig,
    logger: Any,
    trial: optuna.trial.Trial | None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk‑forward splits based on config.
    Returns a list of (train, test) DataFrames.
    """
    if config.cv_folds > 1:
        splits = get_walk_forward_splits(raw_df, n_folds=config.cv_folds)
    else:
        n_test = round(len(raw_df) * config.test_fraction)
        n_train = len(raw_df) - n_test
        splits = [(raw_df.iloc[:n_train], raw_df.iloc[n_test:])]
    n_splits = len(splits)
    print_if_not_trial(
        logger,
        logging.DEBUG,
        trial,
        f"Dataset split into {n_splits}-fold walk-forward cross-validation.",
    )
    return splits
