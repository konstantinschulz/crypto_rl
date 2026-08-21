# Workspace Rules: Crypto RL Agent Development

You are working in the `crypto_rl` workspace. Only the new, refactored setup must be used for any code edits, execution, and research.

## Guidelines & Behavioral Constraints

1. **Active Files & Structure:**
   - **Do NOT** use, reference, edit, or run the legacy `minimal_rl.py` file or its copies in `docs/archive/`.
   - The primary entry point for training and evaluation is [main.py](file:///home/konstantin/dev/crypto_rl/main.py).
   - Core source code (environment, CLI arguments, callbacks, data loading, etc.) is located inside the [crypto_rl](file:///home/konstantin/dev/crypto_rl/crypto_rl/) subfolder. Specifically:
     - [crypto_rl/env.py](file:///home/konstantin/dev/crypto_rl/crypto_rl/env.py) implements the gym environment `MinimalCryptoEnv`.
     - [crypto_rl/cli.py](file:///home/konstantin/dev/crypto_rl/crypto_rl/cli.py) defines the command-line arguments.
     - [crypto_rl/callbacks.py](file:///home/konstantin/dev/crypto_rl/crypto_rl/callbacks.py) handles the `DashboardCallback`.
     - [crypto_rl/data.py](file:///home/konstantin/dev/crypto_rl/crypto_rl/data.py) provides helpers for loading OHLCV price histories.

2. **Dashboard State & Logging:**
   - Run logs (`*.jsonl`) and the dashboard `state.json` are written to and read from the unified path `logs/run-YYYYMMDD-HHMMSS-minimal/`.
   - Never use `rl_dashboard_runs/`.

3. **Running the Agent & Python Environment:**
   - The primary Python interpreter for this workspace is located at `./.conda/bin/python` (or using `./.conda/bin/python main.py ...`).
   - To train the model or start a run:
     ```bash
     ./.conda/bin/python main.py --dashboard --n-rows 10000 --timesteps 20000
     ```
