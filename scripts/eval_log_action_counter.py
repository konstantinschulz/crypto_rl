import glob
import pandas as pd

f = sorted(glob.glob("logs/run-*/actions_eval_*.parquet"))[-1]
df = pd.read_parquet(f)
from collections import Counter

print(Counter(df["action_type"]))
print("Final portfolio:", df["portfolio"].iloc[-1])
