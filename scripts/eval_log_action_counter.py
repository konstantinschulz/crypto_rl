import glob
from collections import Counter

import optuna
import pandas as pd

from crypto_rl.env.logging_utils import LoggerBase, selective_logger


def eval_log_action_counter(logger: LoggerBase, trial: optuna.Trial | None = None) -> None:
    f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
    df = pd.read_parquet(f)
    logger.log(str(Counter(df["action_type"])), trial)
    logger.log(f"Final portfolio: {df['portfolio'].iloc[-1]}", trial)


if __name__ == "__main__":
    with selective_logger("") as logger:
        eval_log_action_counter(logger)
