# GitHub Workspace Instructions

## Interpreter and environment

This repository uses a project-local Python interpreter in:

`./.conda/bin/python`

Use this interpreter for all scripts and experiments to avoid missing dependency errors.

## Standard command pattern

Prefer these binaries explicitly:

- Python: `./.conda/bin/python`
- Pip: `./.conda/bin/pip`
- Streamlit: `./.conda/bin/streamlit`

Examples:

```bash
./.conda/bin/python -u run_experiment9b.py --phase gate --out artifacts/experiments/results/test.json
./.conda/bin/python -u run_experiment10.py --days 90 --max-symbols 5 --train-steps 25000 --seeds 11,23
./.conda/bin/streamlit run streamlit_dashboard.py --server.port 8766
./.conda/bin/pip install -r requirements.txt
```

## Why this matters

`run_experiment9b.py` builds subprocess commands from `sys.executable`. If the parent process is started with the wrong Python binary, child `rl_trader.py` runs may fail (for example, `ModuleNotFoundError: pandas`).

## Quick check

```bash
./.conda/bin/python --version
./.conda/bin/python -c "import pandas, torch; print('env-ok')"
```

## Long Experiment Execution Protocol (mandatory)

For any experiment that may run longer than a short smoke test:

1. Always persist full stdout/stderr to a log file under `artifacts/experiments/logs/`.
2. Use `-u` (unbuffered Python) so log lines flush continuously.
3. Do not rely on terminal scrollback for results; parse summary lines from the saved log.
4. Do not ask the user to poll/status-check after you started a long run.
5. If you started the run, you must:
   - sleep for a reasonable interval,
   - poll the run/log automatically,
   - continue polling until completion,
   - report fold/case progress and final summary in one response.
6. After **every** experiment, show the results to the experts and get their judgment on the best next step before planning anything else.
7. When the experts select a **new recipe family**, start it **immediately** without asking for permission first.
8. If the experts conclude a family is spent, do **not** start another near-duplicate run; stop and wait for a materially different family.

### Canonical launch pattern

```bash
LOG="artifacts/experiments/logs/<run_name>.log"
./.conda/bin/python -u <script>.py <args...> > "$LOG" 2>&1
```

### Canonical summary extraction pattern

```bash
grep -E 'FOLD [0-9]|Fold [0-9] summary|WALK-FORWARD SUMMARY|Val Trades:|Test Trades:' "$LOG"
```

