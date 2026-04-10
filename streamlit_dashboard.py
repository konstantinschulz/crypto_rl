import json
import re
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RL Trader Dashboard", layout="wide")
st.title("RL Training Dashboard")

# Initialize session state to track selected run and auto-refresh
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "last_index_mtime" not in st.session_state:
    st.session_state.last_index_mtime = 0
if "last_run_mtime" not in st.session_state:
    st.session_state.last_run_mtime = 0
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "finance_view" not in st.session_state:
    st.session_state.finance_view = "Both"
if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None


def load_index():
    """Load index file with file modification time tracking for auto-refresh."""
    index_file = Path("rl_dashboard_index.json")
    if index_file.exists():
        try:
            current_mtime = index_file.stat().st_mtime
            # Update mtime tracking
            st.session_state.last_index_mtime = current_mtime
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_run(state_file):
    """Load run state file without caching to ensure fresh data on run selection change."""
    state_path = Path(state_file)
    if state_path.exists():
        try:
            # Update mtime tracking
            st.session_state.last_run_mtime = state_path.stat().st_mtime
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _to_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _run_epoch_from_id(run_id: str) -> int:
    # run id format: run-YYYYMMDD-HHMMSS-xxxxxx
    match = re.match(r"^run-(\d{8})-(\d{6})-[a-zA-Z0-9]+$", str(run_id or ""))
    if not match:
        return 0
    try:
        dt = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
        return int(dt.timestamp())
    except ValueError:
        return 0


def _parse_dashboard_ts(ts_value):
    if not ts_value or not isinstance(ts_value, str):
        return None
    try:
        return datetime.strptime(ts_value, "%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return None


def _format_elapsed(seconds):
    if seconds is None:
        return "-"
    total = max(0, int(_to_float(seconds, 0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _is_running_status(status):
    return str(status or "").lower() in {"initializing", "running"}


def _elapsed_seconds_for_run(run_meta, now_dt=None):
    if not isinstance(run_meta, dict):
        return None

    now_dt = now_dt or datetime.utcnow()
    status = run_meta.get("status")

    started = _parse_dashboard_ts(run_meta.get("started_at"))
    finished = _parse_dashboard_ts(run_meta.get("finished_at"))

    if _is_running_status(status) and started is not None:
        return max(0, int((now_dt - started).total_seconds()))

    if started is not None and finished is not None:
        return max(0, int((finished - started).total_seconds()))

    elapsed = run_meta.get("elapsed_seconds")
    if elapsed is None:
        return None
    return max(0, int(_to_float(elapsed, 0)))


def _latest_series_value(series_map, key, default=0.0):
    rows = series_map.get(key, []) if isinstance(series_map, dict) else []
    if not isinstance(rows, list):
        return float(default)
    for row in reversed(rows):
        if isinstance(row, dict):
            value = row.get("value")
            if value is not None:
                return _to_float(value, default)
    return float(default)


def _latest_nonzero_series_value(series_map, key, default=0.0):
    rows = series_map.get(key, []) if isinstance(series_map, dict) else []
    if not isinstance(rows, list):
        return float(default)
    for row in reversed(rows):
        if isinstance(row, dict):
            value = _to_float(row.get("value"), 0.0)
            if abs(value) > 1e-12:
                return float(value)
    return float(default)


def _empirical_y_domain(df: pd.DataFrame, pad_ratio: float = 0.08):
    if df is None or df.empty:
        return None
    values = pd.to_numeric(pd.Series(df.to_numpy().ravel()), errors="coerce").dropna()
    if values.empty:
        return None

    vmin = float(values.min())
    vmax = float(values.max())

    if vmax == vmin:
        pad = max(abs(vmin) * 0.02, 1e-6)
        return [vmin - pad, vmax + pad]

    pad = (vmax - vmin) * float(pad_ratio)
    return [vmin - pad, vmax + pad]


def _resolve_dashboard_kpis(data):
    run = data.get("run", {}) if isinstance(data, dict) else {}
    finance = data.get("finance", {}) if isinstance(data, dict) else {}
    splits = data.get("splits", {}) if isinstance(data, dict) else {}
    series = data.get("series", {}) if isinstance(data, dict) else {}

    final_summary = run.get("final_summary") if isinstance(run.get("final_summary"), dict) else {}
    
    finance_trades = int(_to_float(finance.get("trades", 0), 0))
    final_trades = int(_to_float(final_summary.get("trades", 0), 0))
    
    series_trades_list = series.get("trades", [])
    max_series_trades = 0
    if isinstance(series_trades_list, list) and len(series_trades_list) > 0:
        values = [int(_to_float(item.get("value", 0), 0)) for item in series_trades_list if isinstance(item, dict)]
        max_series_trades = max(values) if values else 0

    trades = max(finance_trades, final_trades, max_series_trades)

    realized_pnl = _to_float(final_summary.get("realized_pnl", 0), 0)
    if realized_pnl == 0.0 and _to_float(finance.get("realized_pnl", 0), 0) != 0.0:
        realized_pnl = _to_float(finance.get("realized_pnl", 0), 0)
    
    if realized_pnl == 0.0:
        realized_pnl = _latest_series_value(series, "realized_pnl", 0.0)

    win_rate_pct = _to_float(final_summary.get("win_rate_pct", 0), 0)
    if win_rate_pct == 0.0 and _to_float(finance.get("win_rate_pct", 0), 0) != 0.0:
        win_rate_pct = _to_float(finance.get("win_rate_pct", 0), 0)
    if win_rate_pct == 0.0 and trades > 0:
        # Some runs append a terminal eval snapshot with win_rate=0; use latest non-zero historical value.
        win_rate_pct = _latest_nonzero_series_value(series, "win_rate", 0.0)

    portfolio_value = _to_float(final_summary.get("portfolio_value", 0), 0)
    if portfolio_value == 0.0 or portfolio_value == 100.0:
        finance_pv = _to_float(finance.get("portfolio_value", 0), 0)
        if finance_pv != 0.0 and finance_pv != 100.0:
            portfolio_value = finance_pv
    if portfolio_value == 0.0:
        portfolio_value = _latest_series_value(series, "portfolio_value", 0.0)

    return {
        "trades": trades,
        "realized_pnl": realized_pnl,
        "win_rate_pct": win_rate_pct,
        "portfolio_value": portfolio_value,
        "train_realized_pnl": _latest_series_value(series, "realized_pnl", 0.0),
        "train_portfolio_value": _latest_series_value(series, "portfolio_value", 0.0),
    }


index_data = load_index()
runs = index_data.get("runs", [])
if not runs:
    st.warning("No runs found yet.")
    st.stop()

runs = sorted(runs, key=lambda r: (_run_epoch_from_id(r.get("run_id", "")), str(r.get("run_id", ""))), reverse=True)
runs_by_id = {r.get("run_id", "unknown"): r for r in runs}

run_opts = [r.get("run_id", "unknown") for r in runs]
default_idx = 0
latest_model_run = index_data.get("latest_model_run")
if isinstance(latest_model_run, dict):
    latest_model_id = latest_model_run.get("run_id")
    for i, r in enumerate(runs):
        if r.get("run_id") == latest_model_id:
            default_idx = i
            break

if st.session_state.selected_run_id in run_opts:
    selected_idx = run_opts.index(st.session_state.selected_run_id)
else:
    selected_idx = default_idx
    st.session_state.selected_run_id = run_opts[selected_idx]

def _run_label(run_id):
    run_meta = runs_by_id.get(run_id, {})
    mode = run_meta.get("mode", "-")
    status = run_meta.get("status", "-")
    elapsed_label = _format_elapsed(_elapsed_seconds_for_run(run_meta))
    return f"{run_id} | {mode} | {status} | {elapsed_label}"


sel_run_id = st.sidebar.selectbox(
    "Select Run",
    run_opts,
    index=selected_idx,
    format_func=_run_label,
    key="selected_run_id",
)
st.session_state.finance_view = st.sidebar.selectbox(
    "Finance View",
    ["Both", "Train", "Eval"],
    index=["Both", "Train", "Eval"].index(st.session_state.finance_view)
    if st.session_state.finance_view in {"Both", "Train", "Eval"}
    else 0,
)

# Track run selection changes
if sel_run_id != st.session_state.current_run_id:
    st.session_state.current_run_id = sel_run_id

sel_run = next((r for r in runs if r.get("run_id") == sel_run_id), runs[0])
state_file = sel_run.get("state_file", "rl_dashboard_state.json")
data = load_run(state_file)

if not data:
    st.error(f"Could not load state file {state_file}")
    st.stop()

run = data.get("run", {})
tech = data.get("technical", {})
series = data.get("series", {})
kpis = _resolve_dashboard_kpis(data)

st.sidebar.markdown(f"**Mode:** {run.get('mode', '-')}")
st.sidebar.markdown(f"**Status:** {run.get('status', '-')}")
st.sidebar.markdown(f"**Started:** {run.get('started_at', '-')}")
st.sidebar.markdown(f"**Elapsed:** {_format_elapsed(_elapsed_seconds_for_run(run))}")
st.sidebar.markdown(f"**Progress:** {run.get('progress_pct', 0)}%")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Data Intervals", f"{int(_to_float(tech.get('num_data_rows', 0), 0)):,}")
col2.metric("Eval Trades", f"{int(kpis['trades'])}")
col3.metric("Eval Portfolio", f"{kpis['portfolio_value']:.2f}")
col4.metric("Eval Realized PnL", f"{kpis['realized_pnl']:.2f}")
col5.metric("Train Realized PnL", f"{kpis['train_realized_pnl']:.2f}")
col6.metric("Eval Win Rate", f"{kpis['win_rate_pct']:.2f}%")

st.subheader("Training Series")

# Portfolio Value
st.markdown("### Portfolio Value")
portfolio_series = {}
df_port_train = pd.DataFrame(series.get("portfolio_value", []))
df_port_dev = pd.DataFrame(series.get("dev_portfolio_value", []))
df_port_test = pd.DataFrame(series.get("test_portfolio_value", []))
if not df_port_train.empty and "step" in df_port_train.columns:
    # Legacy runs may have split snapshots appended at the same step; keep first to preserve train trace.
    df_port_train.drop_duplicates(subset=["step"], keep="first", inplace=True)
    df_port_train.set_index("step", inplace=True)
    portfolio_series["Train"] = df_port_train["value"]
if not df_port_dev.empty and "step" in df_port_dev.columns:
    df_port_dev.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_port_dev.set_index("step", inplace=True)
    portfolio_series["Dev"] = df_port_dev["value"]
if not df_port_test.empty and "step" in df_port_test.columns:
    df_port_test.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_port_test.set_index("step", inplace=True)
    portfolio_series["Test"] = df_port_test["value"]

if portfolio_series:
    portfolio_df = pd.DataFrame(portfolio_series).dropna(how="all")
    # Interpolate using index and fill edges so sparse Eval points draw as constant lines
    portfolio_df = portfolio_df.interpolate(method="index").ffill().bfill()
    
    view = st.session_state.finance_view
    if view == "Train":
        keep_cols = [c for c in portfolio_df.columns if c == "Train"]
        portfolio_df = portfolio_df[keep_cols] if keep_cols else pd.DataFrame()
    elif view == "Eval":
        keep_cols = [c for c in portfolio_df.columns if c in {"Dev", "Test"}]
        portfolio_df = portfolio_df[keep_cols] if keep_cols else pd.DataFrame()

    if not portfolio_df.empty:
        y_domain = _empirical_y_domain(portfolio_df)
        portfolio_long = portfolio_df.reset_index().melt(
            id_vars=["step"],
            var_name="Series",
            value_name="value",
        )
        chart = (
            alt.Chart(portfolio_long)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y(
                    "value:Q",
                    title="Portfolio Value",
                    scale=alt.Scale(domain=y_domain, zero=False, nice=False),
                ),
                color=alt.Color("Series:N", title="Series"),
            )
        )
        st.altair_chart(chart, use_container_width=True)

# Realized PnL
st.markdown("### Realized PnL")
pnl_series = {}
df_pnl_train = pd.DataFrame(series.get("realized_pnl", []))
df_pnl_dev = pd.DataFrame(series.get("dev_realized_pnl", []))
df_pnl_test = pd.DataFrame(series.get("test_realized_pnl", []))
if not df_pnl_train.empty and "step" in df_pnl_train.columns:
    # Keep first for legacy compatibility where terminal split points may share the train step.
    df_pnl_train.drop_duplicates(subset=["step"], keep="first", inplace=True)
    df_pnl_train.set_index("step", inplace=True)
    pnl_series["Train"] = df_pnl_train["value"]
if not df_pnl_dev.empty and "step" in df_pnl_dev.columns:
    df_pnl_dev.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_pnl_dev.set_index("step", inplace=True)
    pnl_series["Dev"] = df_pnl_dev["value"]
if not df_pnl_test.empty and "step" in df_pnl_test.columns:
    df_pnl_test.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_pnl_test.set_index("step", inplace=True)
    pnl_series["Test"] = df_pnl_test["value"]

if pnl_series:
    pnl_df = pd.DataFrame(pnl_series).dropna(how="all")
    pnl_df = pnl_df.interpolate(method="index").ffill().bfill()
    
    view = st.session_state.finance_view
    if view == "Train":
        keep_cols = [c for c in pnl_df.columns if c == "Train"]
        pnl_df = pnl_df[keep_cols] if keep_cols else pd.DataFrame()
    elif view == "Eval":
        keep_cols = [c for c in pnl_df.columns if c in {"Dev", "Test"}]
        pnl_df = pnl_df[keep_cols] if keep_cols else pd.DataFrame()

    st.line_chart(pnl_df)

st.markdown("### Rewards")
df_train_r = pd.DataFrame(series.get("train_reward", []))
df_test_r = pd.DataFrame(series.get("test_reward", []))
df_dev_r = pd.DataFrame(series.get("dev_reward", []))

rewards_dict = {}
if not df_train_r.empty and "step" in df_train_r.columns:
    df_train_r.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_train_r.set_index("step", inplace=True)
    rewards_dict["Train"] = df_train_r["value"]
if not df_test_r.empty and "step" in df_test_r.columns:
    df_test_r.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_test_r.set_index("step", inplace=True)
    rewards_dict["Test"] = df_test_r["value"]
if not df_dev_r.empty and "step" in df_dev_r.columns:
    df_dev_r.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_dev_r.set_index("step", inplace=True)
    rewards_dict["Dev"] = df_dev_r["value"]

if rewards_dict:
    r_df = pd.DataFrame(rewards_dict).dropna(how="all")
    r_df = r_df.interpolate(method="index").ffill().bfill()
    st.line_chart(r_df)

st.markdown("### Loss")
df_tl = pd.DataFrame(series.get("train_loss", []))
df_pl = pd.DataFrame(series.get("policy_loss", []))
df_vl = pd.DataFrame(series.get("value_loss", []))

loss_dict = {}
if not df_tl.empty and "step" in df_tl.columns:
    df_tl.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_tl.set_index("step", inplace=True)
    loss_dict["Train"] = df_tl["value"]
if not df_pl.empty and "step" in df_pl.columns:
    df_pl.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_pl.set_index("step", inplace=True)
    loss_dict["Policy"] = df_pl["value"]
if not df_vl.empty and "step" in df_vl.columns:
    df_vl.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_vl.set_index("step", inplace=True)
    loss_dict["Value"] = df_vl["value"]

if loss_dict:
    l_df = pd.DataFrame(loss_dict).dropna(how="all")
    l_df = l_df.interpolate(method="index").ffill().bfill()
    st.line_chart(l_df)

st.markdown("### Memory Usage (MB)")
memory_series = {}
df_ram = pd.DataFrame(series.get("ram_mb", []))
df_vram_alloc = pd.DataFrame(series.get("vram_allocated_mb", []))
df_vram_reserved = pd.DataFrame(series.get("vram_reserved_mb", []))
df_vram_process = pd.DataFrame(series.get("vram_process_mb", []))
df_vram_gpu_used = pd.DataFrame(series.get("vram_gpu_used_mb", []))

if not df_ram.empty and "step" in df_ram.columns:
    df_ram.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_ram.set_index("step", inplace=True)
    memory_series["RAM"] = df_ram["value"]

if not df_vram_alloc.empty and "step" in df_vram_alloc.columns:
    df_vram_alloc.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_vram_alloc.set_index("step", inplace=True)
    memory_series["VRAM Allocated"] = df_vram_alloc["value"]

if not df_vram_reserved.empty and "step" in df_vram_reserved.columns:
    df_vram_reserved.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_vram_reserved.set_index("step", inplace=True)
    memory_series["VRAM Reserved"] = df_vram_reserved["value"]

if not df_vram_process.empty and "step" in df_vram_process.columns:
    df_vram_process.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_vram_process.set_index("step", inplace=True)
    memory_series["VRAM Process (nvidia-smi)"] = df_vram_process["value"]

if not df_vram_gpu_used.empty and "step" in df_vram_gpu_used.columns:
    df_vram_gpu_used.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_vram_gpu_used.set_index("step", inplace=True)
    memory_series["VRAM GPU Used (nvidia-smi)"] = df_vram_gpu_used["value"]

if memory_series:
    memory_df = pd.DataFrame(memory_series).dropna(how="all")
    if not memory_df.empty:
        y_domain = _empirical_y_domain(memory_df)
        memory_long = memory_df.reset_index().melt(
            id_vars=["step"],
            var_name="Series",
            value_name="value",
        )
        memory_chart = (
            alt.Chart(memory_long)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y(
                    "value:Q",
                    title="Memory (MB)",
                    scale=alt.Scale(domain=y_domain, zero=False, nice=False),
                ),
                color=alt.Color("Series:N", title="Series"),
            )
        )
        st.altair_chart(memory_chart, use_container_width=True)

# Auto-refresh mechanism: check for file updates and rerun if needed
st.sidebar.divider()
st.sidebar.subheader("💾 Auto-Refresh")
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.session_state.auto_refresh_enabled = st.checkbox(
        "Enable auto-refresh",
        value=st.session_state.auto_refresh_enabled,
        help="Automatically update dashboard every 2 seconds when enabled"
    )
with col2:
    if st.button("↻", help="Refresh now", use_container_width=True):
        st.rerun()

if st.session_state.auto_refresh_enabled:
    # Use an empty placeholder so we can show updates in the sidebar
    status_msg = st.sidebar.empty()
    status_msg.caption("🔄 Monitoring for changes...")
    
    # Enter polling loop at the very end of the file.
    # At this point, Streamlit has already sent the UI to the browser.
    # We sleep for exactly 2 seconds then check disk.
    # The user can still interact - any interaction drops this thread and restarts.
    index_file = Path("rl_dashboard_index.json")
    run_file = Path(state_file)
    
    while True:
        time.sleep(2)  # Wait 2 seconds before checking
        
        changed = False
        
        # Check if index file was modified
        if index_file.exists():
            try:
                if index_file.stat().st_mtime > st.session_state.last_index_mtime:
                    changed = True
            except Exception:
                pass
                
        # Check if run data file was modified
        if not changed and run_file.exists():
            try:
                if run_file.stat().st_mtime > st.session_state.last_run_mtime:
                    changed = True
            except Exception:
                pass
                
        # Trigger rerun if files changed!
        if changed:
            status_msg.success("🔄 Updating data...", icon="📊")
            st.rerun()
