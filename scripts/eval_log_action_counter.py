import glob
from collections import Counter

import pandas as pd

from crypto_rl.env.logging_utils import LoggerBase, selective_logger


def eval_log_action_counter(logger: LoggerBase):
    f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
    df = pd.read_parquet(f)

    logger.info(str(Counter(df["action_type"])))
    logger.info(f"Final portfolio: {df['portfolio'].iloc[-1]}")


if __name__ == "__main__":
    with selective_logger("") as logger:
        eval_log_action_counter(logger)
