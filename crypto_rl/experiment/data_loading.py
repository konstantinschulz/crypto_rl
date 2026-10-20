# -*- coding: utf-8 -*-
"""Data loading utilities for experiment."""

import logging

import optuna
import pandas as pd

from crypto_rl.config import RLConfig
from crypto_rl.data import FoldSplit, make_walk_forward_splits, read_n_rows
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
