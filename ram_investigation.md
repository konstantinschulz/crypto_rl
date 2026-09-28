# RAM Investigation: Optuna Workers

## Why Tracemalloc Shows ~500 MB but htop Shows ~6 GB

Tracemalloc only instruments **Python's own `malloc` heap**. It is blind to four major memory consumers:

| Consumer | Size | Why Tracemalloc Misses It |
| --- | --- | --- |
| numpy `memmap` arrays (backed by OS file pages) | ~1–2 GB per worker | Backed by `mmap()` syscall, not `malloc` |
| PyArrow C++ allocator (during parquet read) | ~1–2 GB transient | C++ `jemalloc`/system allocator, not Python heap |
| PyTorch C++ allocator (model + rollout buffer) | ~0.5–1 GB | C++ allocator |
| **glibc malloc arena retention** | ~1–2 GB | Memory freed to glibc but **not returned to OS** |

> [!NOTE]
> `malloc_trim(0)` is already called in `fold_runner.py` after the cache-build phase, which helps — but glibc may still hold on to large arenas from intermediate numpy computation, which are never returned even after `gc.collect()`.

---

## Root Causes (in order of severity)

### 1. 🔴 `n_rows=18_000_000` — Dominant cause

[`optuna_search.py` line 118](file:///home/konstantin/dev/crypto_rl/scripts/optuna_search.py#L118) hardcodes `n_rows=18000000`. This is the **entire 4-year dataset** (9 symbols × ~2M timestamps). Each worker builds its own full-dataset arrays:

| n_rows | Peak during computation | Steady-state mmap |
| --- | --- | --- |
| 2,000,000 | ~0.5 GB | ~0.1 GB |
| 5,000,000 | ~1.3 GB | ~0.3 GB |
| 10,000,000 | ~2.6 GB | ~0.6 GB |
| **18,000,000** | **~4.6 GB** | **~1.0 GB** |

### 2. 🔴 `precalculate_static_obs` — 39+ large intermediate arrays coexist

[`feature_utils.py` `precalculate_static_obs()`](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/feature_utils.py#L40) computes ~39 intermediate `(T, N)` float32 arrays that **all live simultaneously** in memory before the assembly loop `for t in range(W, T)` starts. At `T=2M`, `N=9`:

- Each `(T, N)` float32 array = **69 MB**
- `precalc_static_obs (T+1, 122)` = **931 MB**
- **Total peak: ~3.5 GB** (numpy only) + **~1 GB** for the pandas raw_df

None of the intermediates are `del`-ed before the assembly loop, so they all coexist.

### 3. 🟡 `DummyVecEnv` with `n_envs=9`

Each of the 9 envs holds a Python reference to the **same mmap arrays** (`prices_arr`, `static_obs`, `norm_vol_arr`). The OS correctly shares physical pages for mmap, so this doesn't multiply memory. However, it does mean every page of all three arrays gets touched during training, making the full 1 GB of mmap resident in each worker's RSS.

### 4. 🟡 glibc arena fragmentation after cache build

Even after `del raw_df; gc.collect(); malloc_trim(0)`, glibc may retain large thread-specific arenas. The `malloc_trim` only trims the main arena — thread arenas (used by numpy's OpenBLAS threads) are not trimmed unless you call `malloc_trim` on each thread's heap or use `MALLOC_ARENA_MAX=1`.

### 5. 🟡 PyArrow leaves C++ buffers in memory

`dataset.to_table(...).to_pandas()` creates an Arrow table in C++ memory, then copies it to a pandas DataFrame. During the conversion, both exist simultaneously: ~0.5 GB (Arrow f32) + ~1 GB (pandas f64) = ~1.5 GB.

---

## How to Fix It

### Fix 1: Reduce `n_rows` for Optuna sweeps (immediate, biggest win)

In [`optuna_search.py` line 118](file:///home/konstantin/dev/crypto_rl/scripts/optuna_search.py#L118), change:

```python
n_rows=18000000,  # 40000 / 400000
```

to something like:

```python
n_rows=5_000_000,  # ~1.3 GB peak vs 4.6 GB
```

For hyperparameter search, you rarely need all 4 years of data. 5M rows (~10 months) is typically sufficient to get a reliable signal.

### Fix 2: Free intermediate arrays in `precalculate_static_obs` (medium effort)

In [`feature_utils.py`](file:///home/konstantin/dev/crypto_rl/crypto_rl/env/feature_utils.py#L40), after computing each batch of intermediate arrays and stacking them into `precalc_static_obs`, explicitly delete the intermediates before starting the assembly loop:

```python
# After computing all arrays, stack them all at once (vectorized) instead of a row loop
# Then del the intermediates before returning
del returns, delta, gain, loss, avg_gain, avg_loss, rs, pv, pv_sum, vol_sum, vwap
del mean_W, std_W, mean_3, vol_mean, vol_mean_24h, pv_sum_24h, vol_sum_24h, vwap_24h
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

Also replace the row-by-row Python loop with a single vectorized `np.stack` — this is both faster and uses less transient memory.

### Fix 3: Set `MALLOC_ARENA_MAX=1` to prevent glibc arena bloat

Add to [`optuna_search.py`](file:///home/konstantin/dev/crypto_rl/scripts/optuna_search.py) (at the very top, before any imports):

```python
os.environ["MALLOC_ARENA_MAX"] = "1"  # Prevent glibc from creating many arenas
```

This forces all threads to share a single malloc arena, which `malloc_trim` can actually reclaim.

### Fix 4: Pre-build the cache once, then fork workers

Instead of each worker independently loading the parquet and building caches (which currently already uses locking), ensure the cache is always pre-built **before** launching workers:

```bash
# Pre-build cache with a single process
./.conda/bin/python -c "
from crypto_rl.config import RLConfig
from crypto_rl.experiment.runner import _prebuild_cache
_prebuild_cache(RLConfig(n_rows=18000000, data_seed=42))
"
# Now launch workers — they only load mmaps, never re-build
./scripts/run_optuna_workers.sh 4
```

Once the cache exists, each worker skips the expensive build phase and only does `np.load(mmap_mode='r')`, which is cheap (maps file pages, doesn't copy).

### Fix 5: Use `numpy.lib.format.open_memmap` with shared memory (advanced)

For truly shared static_obs across workers, store the cache as a POSIX shared memory segment (`/dev/shm`) instead of disk files. Workers can map the same physical pages, so 4 workers together use only as much RAM as 1.

---

## Recommended Action Plan to Reach 3–4 Workers

| Priority | Change | Expected Saving |
| --- | --- | --- |
| **1 (do first)** | `n_rows=5_000_000` in `optuna_search.py` | 4.6 → 1.3 GB peak |
| **2** | `MALLOC_ARENA_MAX=1` env var | 1–2 GB glibc retention |
| **3** | Vectorize assembly loop + del intermediates | 0.5–1 GB peak |
| **4** | Pre-build cache before launching workers | Eliminates cache-build peak entirely |

With fixes 1+2, each worker should drop from ~6 GB to ~2–3 GB RSS, which should allow 4 workers on most machines.
