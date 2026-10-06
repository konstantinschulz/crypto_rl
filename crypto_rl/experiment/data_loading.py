# -*- coding: utf-8 -*-
"""Data loading utilities for experiment."""

import logging

import numpy as np
import optuna
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.data import get_walk_forward_splits, read_n_rows
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


def prepare_splits(
    raw_df: pd.DataFrame,
    config: RLConfig,
    logger: LoggerBase,
    trial: optuna.trial.Trial | None,
) -> list[tuple[int, int, int, str, str]]:
    """Create walk‑forward splits based on config.
    Returns a list of (t_train_max, t_test_min, t_test_max, train_start_str, train_end_str) bounds.
    """
    if config.cv_folds > 1:
        splits = get_walk_forward_splits(raw_df, n_folds=config.cv_folds)
    else:
        unique_times = np.sort(raw_df["open_time"].unique())
        n_test_times = round(len(unique_times) * config.test_fraction)
        t_train_min_raw = unique_times[0]
        t_train_max_raw = unique_times[-(n_test_times + 1)]
        t_test_min_raw = unique_times[-n_test_times]
        t_test_max_raw = unique_times[-1]
        train_start_str = (
            pd.to_datetime(t_train_min_raw)
            .tz_localize("UTC")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        train_end_str = (
            pd.to_datetime(t_train_max_raw)
            .tz_localize("UTC")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        splits = [
            (
                int(t_train_max_raw),
                int(t_test_min_raw),
                int(t_test_max_raw),
                train_start_str,
                train_end_str,
            )
        ]
    n_splits = len(splits)
    logger.print_if_not_trial(
        logging.DEBUG,
        trial,
        f"Dataset split into {n_splits}-fold walk-forward cross-validation.",
    )
    return splits
