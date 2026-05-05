# crypto_rl (Fresh Start)
This repository was reset to focus on a new reward-tuning workflow.
## Kept in root
- Training/runtime core: `rl_trader.py`, `rl_trading_env.py`
- Dashboard: `streamlit_dashboard.py`, `rl_dashboard_server.py`, `rl_dashboard_runs/`, `rl_dashboard_index.json`
- Config/deps: `.env`, `.streamlit/`, `.gitignore`, `requirements.txt`
## Archived material
Historical experiment scripts, logs, Markdown documentation, and helper scripts were moved to:
- `archive/2026-04-27_fresh_start/`
Use this archive for reference while designing new experiments.

## Quickstart: dashboard & minimal runs

Prerequisites:
- This repository contains a local conda environment under `./.conda`. The Python interpreter and tools live in `./.conda/bin`.
- To make the local environment's `python`, `pip` and `streamlit` available in your current shell session, either activate it in your usual way or add it to PATH for the session:

```bash
# from repository root
export PATH="$PWD/.conda/bin:$PATH"
python -m pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

Start a minimal training run from the shell (the trainer will write run state into `rl_dashboard_runs/` and update `rl_dashboard_index.json` so the dashboard can detect it):

```bash
python minimal_rl.py --dashboard --rows 10000 --timesteps 20000 --run-dir rl_dashboard_runs
```

You can also start minimal runs from the dashboard sidebar using "Quick Start: Minimal Train".

Where to look for results:
- Run entries and their live state files: `rl_dashboard_runs/run-*/state.json`
- Global index: `rl_dashboard_index.json`
- Archived experiments and logs: `archive/2026-04-27_fresh_start/`

Stopping runs:
- If you need to stop a background run started from the dashboard or shell, find its PID (e.g. `ps aux | grep minimal_rl.py`) and `kill <PID>`.

Notes:
- The dashboard polls `rl_dashboard_index.json` and the per-run `state.json` files for updates. If runs don't appear immediately, click "Refresh Index" in the dashboard sidebar.
- For large datasets, `minimal_rl.py` reads only the last N rows (configurable via `--rows`) to avoid memory overflows. Consider reducing `--rows` for low-memory machines.


