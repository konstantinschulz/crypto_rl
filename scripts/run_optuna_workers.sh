#!/usr/bin/env bash
# Usage: ./scripts/run_optuna_workers.sh [NUM_WORKERS] [OPTUNA_ARGS...]
# Example: ./scripts/run_optuna_workers.sh 8 --n-trials 20
#
# Workers train on CPU (optuna_search.py). Measured 2026-10-22 on the
# i7-12800H: total throughput keeps rising up to 8 workers (2,262 steps/s vs
# 1,529 with 4), and 8 workers peak at ~7 GB RAM (~0.8 GB private each plus
# the shared memmap caches). Use 6 or fewer if you need more RAM headroom.

NUM_WORKERS=${1:-8}  # Default to 8 workers if not specified
shift 1 2>/dev/null || true

PIDS=()

# Cleanly stop all spawned background processes on Ctrl+C
cleanup() {
    echo -e "\n[!] Stopping all $NUM_WORKERS Optuna workers..."
    
    # Send SIGTERM to all workers (background processes respond to this, unlike SIGINT)
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # Wait for them to exit naturally
    wait 2>/dev/null
    
    # Safety net: If any processes stubbornly survived, force kill them
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    echo "[!] All workers terminated cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Fix 4: Pre-build the memmap cache with a single process BEFORE launching
# workers.  Once the .npy files exist on disk every worker skips the heavy
# numpy feature-computation phase and only does cheap mmap loads, so no
# worker ever hits the ~4 GB peak RAM spike during cache construction.
# ---------------------------------------------------------------------------
./.conda/bin/python scripts/prebuild_cache.py

if [ $? -ne 0 ]; then
    echo "[!] Cache pre-build failed. Aborting."
    exit 1
fi

echo "Starting $NUM_WORKERS parallel Optuna workers with stagger..."

for i in $(seq 1 "$NUM_WORKERS"); do
    ./.conda/bin/python scripts/optuna_search.py "$@" &
    PID=$!
    PIDS+=("$PID")
    echo " -> Worker $i launched (PID $PID)"
    
    # Stagger the next worker to prevent race conditions during DB reads
    if [ "$i" -lt "$NUM_WORKERS" ]; then
        echo "    Waiting 5 seconds for worker to lock its trial..."
        sleep 5
    fi
done

echo -e "\nAll $NUM_WORKERS workers running. Press Ctrl+C to stop all workers.\n"
wait