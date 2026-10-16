import os
import pstats

parent_dir: str = os.path.abspath(".")
profile_stats_file: str = os.path.join(parent_dir, "profile_stats.prof")

if __name__ == "__main__":
    stats = pstats.Stats(profile_stats_file)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)
