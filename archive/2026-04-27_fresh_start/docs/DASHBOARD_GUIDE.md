# Streamlit Dashboard Guide

The Streamlit dashboard provides real-time monitoring of RL training runs with automatic run detection and metric updates.

## Quick Start

### 1. Start the Dashboard
```bash
streamlit run streamlit_dashboard.py --server.port 8766
```
Dashboard opens at `http://localhost:8766`

### 2. Run Training
In a separate terminal:
```bash
python rl_trader.py --days 7 --train-steps 50000 --mode train
```

### 2b. Run the profile-based heavy launcher (recommended for long runs)
```bash
python run_heavy_training.py --profile smoke --dashboard
python run_heavy_training.py --profile first_long --dashboard
```

Profiles in `run_heavy_training.py`:
- `smoke`: quick wiring test (few thousand steps)
- `first_long`: first serious long run
- `first_long_recovery`: improved long-run config after no-trade collapse (recommended)
- `full_long`: larger follow-up run once `first_long` is stable

Every launch now writes persistent artifacts:
- `artifacts/experiments/launches/<timestamp>_<profile>.json` (exact command + overrides)
- `artifacts/experiments/logs/<timestamp>_<profile>.log` (full stdout/stderr)
- `artifacts/experiments/summaries/<timestamp>_<profile>.json` (return code, elapsed time, dashboard final summary)

### 3. View Results
- Dashboard automatically detects the new run
- Metrics update every 2 seconds
- No page reload needed

## How It Works

### Architecture

```
rl_trader.py (Training/Evaluation)
    ↓
RLDashboardWriter (writes every 50-100 steps)
    ↓
rl_dashboard_runs/<run_id>.json   ← Per-run state file
rl_dashboard_index.json           ← Index of all runs
rl_dashboard_state.json           ← Latest state (backward compat)
    ↓
streamlit_dashboard.py (reads every 2 seconds)
    ↓
http://localhost:8766 (web interface updates automatically)
```

### Run ID Format
Each run gets a unique ID: `run-YYYYMMDD-HHMMSS-xxxxxx`
- Date & time when run started
- 6-character hex suffix for uniqueness
- Example: `run-20260327-143200-a9d770`

### File Structure
```
rl_dashboard_index.json
├── runs: [{
│   ├── run_id: "run-20260327-143200-a9d770"
│   ├── mode: "train"
│   ├── status: "running"
│   ├── started_at: "2026-03-27 14:32:00 UTC"
│   ├── finished_at: null
│   ├── elapsed_seconds: 120
│   └── state_file: "rl_dashboard_runs/run-20260327-143200-a9d770.json"
│ ]
├── latest_by_mode: {
│   ├── train: {...}
│   ├── backtest: {...}
│   └── eval: {...}
│ }
└── latest_model_run: {...}

rl_dashboard_runs/run-20260327-143200-a9d770.json
├── ts: "2026-03-27 14:35:00 UTC"
├── run: {
│   ├── run_id: "run-20260327-143200-a9d770"
│   ├── mode: "train"
│   ├── status: "running"
│   ├── progress_pct: 42.5
│   └── ...
│ }
├── technical: {
│   ├── step: 21250
│   ├── learning_rate: 0.0003
│   └── ...
│ }
├── finance: {
│   ├── portfolio_value: 12450.50
│   ├── realized_pnl: 2450.50
│   ├── trades: 15
│   └── win_rate_pct: 60.0
│ }
├── splits: {
│   ├── dev: {...}
│   └── test: {...}
│ }
└── series: {
    ├── portfolio_value: [{step: 100, value: 10050}, ...]
    ├── realized_pnl: [{step: 100, value: 50}, ...]
    ├── train_reward: [{step: 100, value: 0.5}, ...]
    └── ...
  }
```

## Dashboard Interface

### Top Bar (Key Performance Indicators)
```
Step: 21250    Train Loss: 0.01234    Train Reward: 0.1050    Train Portfolio: 101.82
Train PnL (R/U): 0.78 / 1.04    Eval Portfolio: 103.15    Eval PnL: 2.43
```
- **Step**: Current training step (updated every 2 seconds)
- **Train Loss / Reward**: live learning quality indicators
- **Train Portfolio / PnL (R/U)**: live trading equity and realized/unrealized PnL
- **Eval Portfolio / PnL**: latest evaluated split outcome

### Left Sidebar
```
Select Run: [run-20260327-143200-a9d770 ▼]

Mode: train
Status: running
Started: 2026-03-27 14:32:00 UTC
Progress: 42%
```
- **Select Run**: Dropdown to switch between recent runs
- **Mode**: Type of run (train/backtest/eval/backtest_3way)
- **Status**: Current state (initializing/running/completed/failed)
- **Started**: When the run began
- **Progress**: Percentage complete (based on total_timesteps)

### Main Charts

#### Portfolio vs PnL
Line chart showing:
- Portfolio Value (blue line) — Total account worth over time
- Realized PnL (orange line) — Profit/loss from closed trades

Use this to:
- See if agent is making profitable trades
- Identify when drawdowns occur
- Compare portfolio growth vs actual profit

#### Rewards
Line chart showing average reward per step:
- **Train** (blue) — Reward during training phase
- **Dev** (orange) — Reward during development/validation
- **Test** (green) — Reward during testing on unseen data

Use this to:
- Track if agent is learning (Train reward should increase)
- Compare train vs test performance
- Detect overfitting (Train >> Test = overfit)

#### Loss
Line chart showing loss metrics:
- **Train** (blue) — Overall training loss
- **Policy** (orange) — Policy gradient loss
- **Value** (green) — Value function loss

Use this to:
- Monitor training stability
- Detect divergence (loss spikes = problems)
- Confirm learning progress (loss should decrease)

#### Additional Monitoring Series
- **Unrealized PnL**: open-position PnL path over training
- **Return / Drawdown (%)**: reward-quality vs risk path
- **PPO Stability (KL / Clip Fraction)**: update step health and clipping pressure

## Usage Patterns

### Pattern 1: Long-Running Training (Recommended)

**Setup:**
```bash
# Terminal 1: Start dashboard once
streamlit run streamlit_dashboard.py --server.port 8766

# Terminal 2: Run any training/eval commands
python rl_trader.py --days 7 --train-steps 50000 --mode train
python rl_trader.py --days 14 --mode backtest_3way  # Different terminal
```

**Benefits:**
- Dashboard stays open all day
- Handles multiple concurrent runs
- Automatic run detection
- No reload needed

### Pattern 2: Monitor Multiple Runs

1. Start dashboard (Terminal 1)
2. Run first training (Terminal 2)
3. See results in dashboard
4. Start second training in new terminal (Terminal 3)
5. Switch between runs using dropdown

Dashboard gracefully handles this — each run writes to its own file.

### Pattern 3: Run-Specific Monitoring

```bash
# Terminal 1: Dashboard
streamlit run streamlit_dashboard.py --server.port 8766

# Terminal 2: Run backtest
python rl_trader.py --days 14 --mode backtest_3way --train-steps 50000
```

- Dashboard auto-selects the new run
- Shows all 3 phases (train/dev/test) metrics
- `splits.dev` and `splits.test` populate during run

### Pattern 4: Evaluate Existing Models

```bash
# Terminal 1: Dashboard
streamlit run streamlit_dashboard.py --server.port 8766

# Terminal 2: Eval run
python rl_trader.py --mode eval --model models/final_model.zip --days 7
```

- Creates new `run_id` for eval
- Shows eval metrics in dashboard
- Useful for comparing models side-by-side

## Interpreting Metrics

### Healthy Training
- Train reward: gradually increasing
- Test reward: increasing (or flat, but not decreasing)
- Losses: decreasing or stable
- Portfolio: trending upward
- Win rate: increasing toward 50%+ over time

### Problematic Training
- Train reward: flat or decreasing
- Test reward: decreasing (overfitting)
- Losses: spiking or diverging
- Portfolio: trending downward
- No trades executed (agent is frozen)

### What to Do
- If losses spike: reduce `learning_rate` (try 1e-4 or 1e-5)
- If portfolio flat: train longer (`--train-steps 100000`)
- If test << train: reduce model complexity (already optimized)
- If no trades: check config (position limits, etc.)

## Common Issues

### Dashboard Shows No Runs
**Cause**: No `rl_dashboard_index.json` file

**Solution**:
```bash
# Run at least one training session
python rl_trader.py --days 3 --train-steps 5000 --mode train
# Then refresh dashboard (Ctrl+Shift+R in browser)
```

### Stale Data in Dashboard
**Cause**: Streamlit cache not clearing

**Solution**:
```bash
# Hard refresh (clears Streamlit cache)
Ctrl + Shift + R

# Or restart dashboard server
pkill -f streamlit
streamlit run streamlit_dashboard.py --server.port 8766
```

### Dashboard Shows Old Runs
**Cause**: Sorted by timestamp, and you're looking at oldest

**Solution**:
- Use the **Select Run** dropdown
- Recent runs appear at top
- Look for "running" status to find active runs

### Metrics Stop Updating
**Cause**: Training process crashed or hung

**Solution**:
```bash
# Check if training process is still running
ps aux | grep rl_trader.py

# If stuck, kill it
pkill -f rl_trader.py

# Restart training
python rl_trader.py --days 7 --train-steps 50000 --mode train
```

## Advanced Usage

### Filter Runs by Mode
The dropdown shows all recent runs. You can mentally filter them:
- Runs with `train` in name: Training runs
- Runs with `backtest` in name: Backtest runs
- Runs with `eval` in name: Model evaluations

### Compare Runs Side-by-Side
1. Open dashboard in two browser tabs
2. Tab 1: Select run A
3. Tab 2: Select run B
4. Compare metrics visually

### Export Run Data
```bash
# Each run's full state is in JSON
cat rl_dashboard_runs/run-20260327-143200-a9d770.json | jq '.series'
# Extract to CSV for further analysis
```

### Modify Dashboard Appearance
Edit `streamlit_dashboard.py` to:
- Add custom metrics
- Change chart types
- Modify KPI display
- Add filtering options

See [RL_TRADING_README.md](RL_TRADING_README.md#add-custom-dashboard-metrics) for examples.

## Extending the Dashboard

### Add a Custom Metric

1. **In rl_trader.py**, write your metric:
```python
dashboard_writer.state['series']['my_metric'].append({
    'step': current_step,
    'value': my_value
})
```

2. **In streamlit_dashboard.py**, display it:
```python
df_custom = pd.DataFrame(series.get('my_metric', []))
if not df_custom.empty and 'step' in df_custom.columns:
    df_custom.set_index('step', inplace=True)
    st.line_chart(df_custom['value'], use_container_width=True)
```

3. **In rl_trader.py**, initialize the series:
```python
self.state['series']['my_metric'] = []
```

### Add a New KPI

```python
# In streamlit_dashboard.py, after existing KPIs:
my_kpi_value = calculate_my_kpi(data)
col6.metric("My KPI", f"{my_kpi_value:.2f}")
```

## Performance Considerations

### Dashboard Memory Usage
- ~50 MB for running dashboard
- Caches with 2-second TTL
- Automatically clears old cache entries

### Training Memory Usage
- Dashboard writes: ~1 MB per 1000 steps
- Max 40 run files kept (oldest auto-deleted)
- Series limited to 500 points (truncated if longer)

### Browser Considerations
- Works in Chrome, Firefox, Safari
- Auto-refresh every 2 seconds (low overhead)
- Handles 40+ runs in dropdown without lag

## Troubleshooting Dashboard Issues

### High Memory (Dashboard Process)
```bash
# Check dashboard memory
ps aux | grep streamlit | grep -v grep

# Restart to clear cache
pkill -f streamlit
streamlit run streamlit_dashboard.py --server.port 8766
```

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8766

# Use different port
streamlit run streamlit_dashboard.py --server.port 8777
```

### Dashboard Very Slow
- Close other browser tabs
- Restart Streamlit server
- Check system RAM (close other apps)
- Try lighter data (fewer symbols, days)

## Tips for Best Experience

1. **Keep dashboard open all day**
   - No overhead when not training
   - Instantly shows new runs

2. **Use keyboard shortcut to auto-refresh**
   - Ctrl+R to refresh in browser
   - F5 for hard refresh

3. **Monitor from separate machine**
   - Put dashboard on one screen
   - Put training on another
   - Or use remote browser (see network config)

4. **Script multiple runs**
   ```bash
   python rl_trader.py --days 3 --train-steps 10000 --mode train
   python rl_trader.py --days 3 --train-steps 10000 --mode train
   python rl_trader.py --days 7 --train-steps 50000 --mode train
   # Dashboard automatically switches to latest
   ```

## FAQ

**Q: Can I access the dashboard from another machine?**
Yes, configure Streamlit to listen on all interfaces:
```bash
streamlit run streamlit_dashboard.py \
  --server.port 8766 \
  --server.address 0.0.0.0
```
Then access from other machine at `http://<your-ip>:8766`

**Q: What happens if I stop the dashboard?**
Dashboard stops, but training continues. Restart dashboard and it will show the latest metrics.

**Q: Can I run the dashboard and training on the same process?**
Yes, use `--dashboard` flag (starts dashboard automatically):
```bash
python rl_trader.py --days 7 --train-steps 50000 --mode train --dashboard
```
But for long training, separate terminals is recommended.

**Q: How do I reset/clear old runs?**
```bash
# Delete run files (careful!)
rm rl_dashboard_runs/*.json
rm rl_dashboard_index.json

# Restart dashboard (it will be empty until new runs)
```

## See Also

- [RL_TRADING_README.md](RL_TRADING_README.md) — Full system documentation
- [MEMORY_EFFICIENT_QUICKSTART.md](MEMORY_EFFICIENT_QUICKSTART.md) — Quick start for limited RAM
- [rl_trader.py](rl_trader.py) — Training script with RLDashboardWriter
- [streamlit_dashboard.py](streamlit_dashboard.py) — Dashboard implementation

