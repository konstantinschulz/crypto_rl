## 2. Alternative Documentation Approaches

The approach of a single Markdown file has limitations for LLM-assisted
analysis. Consider these alternatives or complements:

### Structured YAML/JSON sidecar

Maintain a machine-readable `hyperparameter_rationale.yaml` alongside this
file. Keys would be parameter names; values would contain `current_value`,
`rationale`, `tuning_history`, and `sensitivity` fields. LLMs can ingest
structured YAML more reliably than prose for comparison tasks.

### Auto-generated eval appendix

Add a script (e.g., `scripts/gen_eval_summary.py`) that reads `state.json`
after each run and appends a timestamped eval snapshot to an `EVAL_HISTORY.md`
file. This gives the LLM a longitudinal view of metric evolution across runs —
more actionable than a single snapshot.

### Inline config annotation

Annotate `run_large.sh` with structured comments above each flag group (e.g.,
`# === REWARD SHAPING ===`). A short preprocessing script can extract these
comments and values into a structured LLM prompt, keeping rationale co-located
with configuration.

### Optuna trial database

The existing `optuna.db` already captures search history. Expose this as a
human-readable trial summary (e.g., via `optuna dashboard` or a CSV export)
and include it as context alongside this document. This gives the LLM ground
truth on which hyperparameter combinations have been tried and their relative
performance.
