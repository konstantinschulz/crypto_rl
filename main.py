import shutil
from pathlib import Path

from crypto_rl.cli import build_parser
from crypto_rl.config import RLConfig
from crypto_rl.experiment import run_experiment


def clear_memmap_cache(config: RLConfig):
    """Safely delete the memory map cache directory."""
    cache_dir = Path(config.data_cache_dir)
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("[Cleanup] Memory map cache cleared.")
        except Exception as e:
            print(f"[Cleanup] Warning: Failed to clear cache: {e}")


def main(config: RLConfig):
    run_experiment(config)


if __name__ == "__main__":
    config: RLConfig = build_parser()
    try:
        main(config)
    finally:
        if config.clear_cache:
            # Guarantees the cache is wiped even if the single run crashes mid-execution
            clear_memmap_cache(config)
