import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RL Trader Dashboard", layout="wide")
st.title("RL Training Dashboard")

# Initialize session state to track selected run
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None


@st.cache_data(ttl=2)
def load_index():
    index_file = Path("rl_dashboard_index.json")
    if index_file.exists():
        try:
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
    }


index_data = load_index()
runs = index_data.get("runs", [])
if not runs:
    st.warning("No runs found yet.")
    st.stop()

runs = sorted(runs, key=lambda r: (_run_epoch_from_id(r.get("run_id", "")), str(r.get("run_id", ""))), reverse=True)

run_opts = [r.get("run_id", "unknown") for r in runs]
default_idx = 0
latest_model_run = index_data.get("latest_model_run")
if isinstance(latest_model_run, dict):
    latest_model_id = latest_model_run.get("run_id")
    for i, r in enumerate(runs):
        if r.get("run_id") == latest_model_id:
            default_idx = i
            break

sel_run_id = st.sidebar.selectbox("Select Run", run_opts, index=default_idx)

# If run selection changed, clear the index cache to get fresh data
if sel_run_id != st.session_state.current_run_id:
    st.session_state.current_run_id = sel_run_id
    load_index.clear()

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
st.sidebar.markdown(f"**Progress:** {run.get('progress_pct', 0)}%")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Step", f"{int(_to_float(tech.get('step', 0), 0))}")
col2.metric("Trades", f"{int(kpis['trades'])}")
col3.metric("Portfolio Value", f"{kpis['portfolio_value']:.2f}")
col4.metric("Realized PnL", f"{kpis['realized_pnl']:.2f}")
col5.metric("Win Rate", f"{kpis['win_rate_pct']:.2f}%")

st.subheader("Training Series")

st.markdown("### Portfolio vs PnL")
df_port = pd.DataFrame(series.get("portfolio_value", []))
df_pnl = pd.DataFrame(series.get("realized_pnl", []))
if not df_port.empty and not df_pnl.empty and "step" in df_port.columns and "step" in df_pnl.columns:
    df_port.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_pnl.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_port.set_index("step", inplace=True)
    df_pnl.set_index("step", inplace=True)
    df_combined = pd.DataFrame({"Portfolio": df_port["value"], "PnL": df_pnl["value"]}).dropna(how="all")
    st.line_chart(df_combined)

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
    st.line_chart(pd.DataFrame(rewards_dict).dropna(how="all"))

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
    st.line_chart(pd.DataFrame(loss_dict).dropna(how="all"))
