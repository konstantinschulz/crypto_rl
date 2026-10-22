"""Optuna workers must not all load the raw data at once (it OOMs the machine)."""

import dataclasses
import multiprocessing as mp
import time

from crypto_rl.config import RLConfig
from crypto_rl.data import FoldSplit
from crypto_rl.env.data_utils import fold_cache_prefixes
from crypto_rl.experiment.data_loading import (
    raw_data_lock,
    read_cached_splits,
    write_cached_splits,
)

DATA_RANGE = (1_600_000_000_000, 1_700_000_000_000)
SPLIT = FoldSplit(
    1_650_000_000_000,
    1_650_000_060_000,
    1_660_000_000_000,
    1_660_000_060_000,
    1_670_000_000_000,
    "2022-04-14 01:20:00 UTC",
    "2022-04-15 08:53:20 UTC",
)


def _hold_lock(cache_dir: str, log, name: str) -> None:
    with raw_data_lock(RLConfig(data_cache_dir=cache_dir)):
        log.put((name, "start", time.monotonic()))
        time.sleep(0.3)
        log.put((name, "end", time.monotonic()))


def test_raw_data_lock_serializes_processes(tmp_path):
    ctx = mp.get_context("spawn")
    log = ctx.Queue()
    procs = [
        ctx.Process(target=_hold_lock, args=(str(tmp_path), log, name))
        for name in ("a", "b", "c")
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    events = sorted((log.get() for _ in range(6)), key=lambda e: e[2])
    # Each holder ends before the next one starts
    for i in range(0, 6, 2):
        assert events[i][1] == "start" and events[i + 1][1] == "end"
        assert events[i][0] == events[i + 1][0]


def test_split_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "crypto_rl.experiment.data_loading.parquet_open_time_range",
        lambda _path: DATA_RANGE,
    )
    config = RLConfig(data_cache_dir=str(tmp_path))
    assert read_cached_splits(config) is None
    write_cached_splits(config, [SPLIT, SPLIT])
    assert read_cached_splits(config) == [SPLIT, SPLIT]
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_prefix_depends_on_window_size():
    config = RLConfig()
    other = dataclasses.replace(config, window_size=config.window_size * 2)
    a = fold_cache_prefixes(config, 9, 0, DATA_RANGE, SPLIT)
    b = fold_cache_prefixes(other, 9, 0, DATA_RANGE, SPLIT)
    assert all(x != y for x, y in zip(a, b))
