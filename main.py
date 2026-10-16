import shutil
from pathlib import Path
from typing import Any

from crypto_rl.cli import build_parser
from crypto_rl.config import RLConfig
from crypto_rl.experiment import run_experiment


def clear_cache(config: RLConfig):
    """Safely delete the memory map and chunks cache directories."""
    for folder in [config.data_cache_dir, config.chunks_dir]:
        cache_dir = Path(folder)
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"[Cleanup] {folder} cache cleared.")
            except Exception as e:
                print(f"[Cleanup] Warning: Failed to clear {folder} cache: {e}")


def main(config: RLConfig, do_profile: bool = False):
    import cProfile
    import io
    import pstats
    from pstats import SortKey

    pr: Any = None
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()
    run_experiment(config)
    if do_profile:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        ps.dump_stats("profile_stats.prof")


if __name__ == "__main__":
    config: RLConfig = build_parser()
    try:
        main(config, do_profile=False)
    finally:
        if config.clear_cache:
            # Guarantees the cache is wiped even if the single run crashes mid-execution
            clear_cache(config)
