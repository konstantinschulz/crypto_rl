# Fresh Start Archive (2026-04-27)
This folder contains pre-reset experiment material moved out of the repository root:
- `scripts/`: historical experiment and tuning runners
- `logs/`: run logs and parsed/manual log files
- `docs/`: legacy Markdown documentation from the root
- `states/`: historical dashboard state snapshots (`rl_dashboard_state_*.json`)
- `artifacts/`: archived `artifacts/experiments`
- `tensorboard/`: archived TensorBoard logs
- `helpers/`: archived non-essential helper scripts (`demo.py`, `dump_all.py`, `dump_logs.py`, `test_env.py`)
Intentionally kept in root for the new reward-tuning phase:
- Dashboard infrastructure: `streamlit_dashboard.py`, `rl_dashboard_server.py`, `rl_dashboard_index.json`, `rl_dashboard_runs/`
- Runtime core: `rl_trader.py`, `rl_trading_env.py`
- Config/environment files: `.env`, `.streamlit/`, `.gitignore`
- Dependency manifest: `requirements.txt`
