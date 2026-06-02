# Artifacts Layout

Generated experiment outputs are stored here to keep repository root clean.

- `experiments/logs/`: run logs (`*.log`)
- `experiments/results/`: result summaries (`*.json`)
- `dashboard/states/`: archived dashboard state snapshots (`rl_dashboard_state_exp*.json`)

Current live dashboard runtime files remain in repository root:
- `rl_dashboard_index.json`
- `rl_dashboard_runs/`

Legacy dashboard server/runtime snapshots were archived under:
- `archive/2026-06-02_legacy_dashboard_server/`
