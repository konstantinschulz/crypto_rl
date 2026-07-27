import glob
import pandas as pd


def eval_log_action_counter():
    f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
    df = pd.read_parquet(f)
    from collections import Counter

    print(Counter(df["action_type"]))
    print("Final portfolio:", df["portfolio"].iloc[-1])


if __name__ == "__main__":
    eval_log_action_counter()
