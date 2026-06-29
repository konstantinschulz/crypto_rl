# crypto_rl (Refactored Setup)
This repository focuses on a reward-tuning workflow for reinforcement learning trading agents.

## Repository Structure
- **Training Entrypoint:** `main.py`
- **Source Code Subfolder:** `crypto_rl/`
  - `crypto_rl/env.py`: Gym trading environment `MinimalCryptoEnv`
  - `crypto_rl/cli.py`: CLI parser definitions
  - `crypto_rl/callbacks.py`: Dashboard logging callbacks
  - `crypto_rl/data.py`: Parquet data ingestion helpers
- **Dashboard:** `streamlit_dashboard.py`, `rl_dashboard_index.json`
- **Config/deps:** `.env`, `.streamlit/`, `.gitignore`, `requirements.txt`

## Archived Material
Historical experiment scripts, logs, Markdown documentation, and legacy/minimal scripts are archived in `docs/archive/` and other subfolders under `docs/`. Use these only for reference.

---

## Quick Start: Dashboard & Training

### Setup (one-time)
```bash
# Activate your conda environment
source .conda/bin/activate

# Install dependencies
pip install -r requirements.txt

# Export path so `streamlit` and `pip` are available in your shell
export PATH="/home/konstantin/dev/crypto_rl/.conda/bin:$PATH"
```

### Start Dashboard
```bash
streamlit run streamlit_dashboard.py
```
The dashboard will open at `http://localhost:8501` and auto-refresh every 2 seconds (toggle in sidebar).

### Start a Training Run

**From shell:**
```bash
python main.py --dashboard --rows 10000 --timesteps 20000
```

**From dashboard:**
1. Open the sidebar → "Quick Start: Minimal Train"
2. Configure rows and timesteps
3. Click "Start Minimal Run"
4. The run will appear in the dropdown; select it to monitor progress

### Dashboard Run State Files
- Index: `rl_dashboard_index.json` (lists all runs)
- Run data: `logs/run-YYYYMMDD-HHMMSS-minimal/state.json` (stored in the same directory as the `*.jsonl` log files)

To stop a run: `Ctrl+C` in the terminal where it started, or kill the process.

---

## Data Streaming & Memory

### The `--rows` Parameter
The `--rows` argument controls how much historical price data is loaded from `binance_spot_1m_last4y_single.parquet`:

```bash
python main.py --rows 10000 --timesteps 20000
```

**Why?** The parquet file is large (~1.3M rows, ~4 years of 1-minute OHLCV data). Loading all of it can cause out-of-memory errors.

**How it works:**
- Uses **pyarrow row-group-aware reading** (if available) to load only the last N row groups instead of the entire file
- Falls back to pandas `.read_parquet(..., columns=['close'])` + tail if pyarrow is unavailable
- Dramatically reduces RAM usage without sacrificing flexibility

**Memory / Resolution Tradeoff:**
| Rows | ~Time Span | RAM Used | Training Fidelity |
|------|-----------|----------|------------------|
| 1,000 | ~16 hours | ~5 MB | Very coarse |
| 5,000 | ~3 days | ~20 MB | Moderate |
| 10,000 | ~7 days | ~40 MB | Good |
| 50,000 | ~35 days | ~200 MB | Excellent |
| 100,000+ | ~70+ days | 400+ MB | Risk of OOM on low-RAM machines |

**Examples:**
```bash
# Minimal test (fast, low memory)
python main.py --rows 1000 --timesteps 5000

# Standard run (7 days of data)
python main.py --dashboard --rows 10000 --timesteps 20000 --run-dir logs

# Extended training (35 days of data, longer convergence time)
python main.py --dashboard --rows 50000 --timesteps 100000 --run-dir logs
```

---

## RL Environment (`crypto_rl/env.py`)

A trading environment for reward-tuning experiments:
- **State:** Last N relative price changes
- **Action:** 0 = Hold cash, 1 = Hold crypto
- **Reward:** Direct portfolio PnL per timestep ($ change)
- **Training:** 80/20 train/test split; evaluate on test set after training

**CLI arguments:**
```
--rows N           Number of last rows from parquet (default: 10000)
--timesteps N      Total PPO training steps (default: 20000)
--dashboard        Enable run-state JSON output for streamlit dashboard
--run-dir PATH     Directory to write run state (default: logs)
```

---

## Notes

- **Python interpreter:** Conda environment at `./.conda/bin/python`
- **Auto-refresh:** The dashboard polls `rl_dashboard_index.json` and run `state.json` every 2 seconds
- **Live metrics:** Training reward and portfolio value are emitted to the dashboard in real-time
- **Stopping runs:** If you started a run in the background, find its PID with `ps aux | grep main.py` and `kill <PID>` to stop it
- **Large datasets:** For low-memory machines, reduce `--rows` to avoid OOM errors. Test with `--rows 1000` first
