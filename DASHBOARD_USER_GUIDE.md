# Dashboard Auto-Refresh: User Guide

## Overview

The RL Trading Dashboard now **automatically updates in real-time** as your training runs progress. No more manual refreshing needed!

## Getting Started

### 1. Start a Training Run with Dashboard

```bash
python rl_trader.py --days 14 --mode backtest --dashboard
```

The script will output:
```
[DASHBOARD] RL dashboard: http://localhost:8766
```

### 2. Open the Dashboard in Your Browser

Navigate to: **http://localhost:8766**

The Streamlit dashboard will load automatically and begin monitoring for updates.

## Dashboard Features

### 📊 Main Display

The dashboard shows real-time metrics:

| Metric | Description |
|--------|-------------|
| **Data Intervals** | Number of price candles being trained on |
| **Trades** | Current number of trades executed |
| **Portfolio Value** | Current value of the trading portfolio |
| **Realized PnL** | Profits/losses from closed positions |
| **Win Rate** | Percentage of winning trades |

### 📈 Charts

Three real-time charts update automatically:

1. **Portfolio vs PnL**
   - Blue line: Portfolio value over time
   - Orange line: Realized profit/loss over time

2. **Rewards**
   - Train: Rewards during training
   - Test: Rewards during backtesting
   - Dev: Rewards during validation

3. **Loss**
   - Train: Training loss curves
   - Policy: Policy loss
   - Value: Value function loss

## Auto-Refresh Controls

### In the Sidebar

```
💾 Auto-Refresh
☑ Enable auto-refresh    ↻
🔄 Monitoring for changes...
```

#### Enable Auto-Refresh Checkbox
- **Checked (✓)** = Dashboard updates every 2 seconds automatically
- **Unchecked ( )** = Dashboard freezes at current state
- **Default**: Checked (ON)

#### Refresh Button (↻)
- Click to immediately refresh the dashboard
- Useful when you want to see latest data right now
- Works whether auto-refresh is on or off

#### Status Indicator
- **🔄 Monitoring for changes...** = Actively polling for updates
- **🔄 Updating data...** = Update detected, refreshing now

### Sidebar Information

```
**Mode:** backtest
**Status:** running
**Started:** 2026-03-30 10:15:23 UTC
**Progress:** 45%
```

Shows current run's:
- Mode (train/backtest/backtest_3way/eval)
- Status (initializing/running/completed/failed)
- Start time
- Progress percentage

## Selecting Different Runs

### Run Dropdown

The "Select Run" dropdown at the top of the sidebar shows all recent runs:

```
Select Run
[run-20260330-101523-a1b2c3 ▼]
```

#### How to Switch Runs
1. Click the dropdown
2. Select a different run from the list
3. Dashboard automatically loads that run's data
4. Charts and metrics update for the new run
5. Auto-refresh continues to monitor the selected run

#### Run List
- **Newest runs first** (most recent at top)
- **Up to 40 recent runs** stored
- **New runs appear automatically** as they start
- **Status indicators** (running/completed/failed)

## Typical Workflow

### Scenario: Start a 14-day backtest

```bash
Terminal 1:
$ python rl_trader.py --days 14 --mode backtest --dashboard
[DASHBOARD] RL dashboard: http://localhost:8766
Starting training...
```

```
Browser: http://localhost:8766

t=0s:   New run appears in "Select Run" dropdown
        Dashboard loads run details
        Status: "initializing"

t=1s:   First metrics appear
        Progress: 0%
        Charts start to populate

t=5s:   Portfolio value updates
        Charts show upward trend
        Trades counter increases
        Auto-refresh: ON (default)

t=30s:  Multiple data points visible in charts
        Training loss decreases
        Portfolio value stabilizes

t=60s:  Significant progress visible
        Progress: 12%
        Multiple trade examples

...continue monitoring...

Run completes:
        Status: "completed"
        Final metrics locked
        Can switch to new run if needed
```

## Best Practices

### 1. Use Auto-Refresh During Training
✅ Keep auto-refresh **ON** during active training runs  
✅ Allows you to monitor progress without action  
✅ Charts update in real-time  

### 2. Toggle Off for Analysis
✅ Turn auto-refresh **OFF** when analyzing a completed run  
✅ Prevents accidental screen refreshes while studying  
✅ Metrics stay stable for screenshot/documentation  

### 3. Manual Refresh for Quick Check
✅ Use **↻ button** to quickly check latest data  
✅ Faster than waiting for automatic 2-second interval  
✅ Useful when you expect critical updates  

### 4. Monitor Multiple Runs
✅ Can watch multiple training runs simultaneously  
✅ Switch between runs in dropdown  
✅ Each run's charts update independently  

## Performance Tips

### Polling Frequency

The dashboard checks for updates every **2 seconds**:

- **Default (2s)**: Good balance of responsiveness and CPU usage
- **Adjust if needed**: Find `time.sleep(2)` in code
  - Smaller value (1s) = More responsive but higher CPU
  - Larger value (5s) = Lower CPU but less responsive

### Browser Performance

- Works best on **modern browsers** (Chrome, Firefox, Safari, Edge)
- **Responsive design** works on tablets and mobile
- **No plugins required**
- Lightweight updates (only file timestamps checked)

### Network Considerations

- **Local only**: No network traffic, purely local file monitoring
- **Works remotely**: If you SSH into the server and port-forward
- **No bandwidth required**: Just checking file modification times

## Troubleshooting

### Dashboard shows no runs
**Problem:** "No runs found yet"

**Solution:**
1. Make sure you started the training with `--dashboard` flag
2. Check that `rl_dashboard_index.json` file exists
3. Wait a few seconds for run to initialize
4. Click ↻ to manually refresh

### Dashboard updates are slow
**Problem:** Updates take more than 2 seconds to appear

**Solution:**
1. Check that "Enable auto-refresh" is checked
2. Try clicking ↻ to verify it refreshes immediately
3. If it's slow, increase your polling interval
4. Check that files are being written (check disk activity)

### Dashboard shows stale data
**Problem:** Data not updating even with auto-refresh on

**Solution:**
1. Click ↻ button to force immediate refresh
2. Toggle auto-refresh off and back on
3. Check if training is actually running
4. Verify `rl_dashboard_runs/` directory has recent files

### Dashboard freezes or crashes
**Problem:** Dashboard becomes unresponsive

**Solution:**
1. Refresh the browser page (F5)
2. Clear browser cache
3. Close and reopen the dashboard
4. Check browser console for errors (F12)
5. Restart Streamlit: `streamlit run streamlit_dashboard.py`

## Keyboard Shortcuts (Browser)

| Shortcut | Action |
|----------|--------|
| `F5` | Refresh page (hard reload) |
| `Ctrl+Shift+R` | Clear cache and reload |
| `F12` | Open developer tools |
| `Ctrl+L` | Focus address bar |

## Mobile/Tablet Usage

The dashboard is **fully responsive**:
- ✅ Works on iPad, Android tablets
- ✅ Touch-friendly controls
- ✅ Readable on smaller screens
- ✅ Auto-refresh works the same way

## Advanced Usage

### Monitoring Metrics

The dashboard tracks these metrics across all training phases:

**Training Phase:**
- Training loss and policy loss
- Rewards during training
- Portfolio value progression
- Trade count and win rate

**Evaluation Phase:**
- Test set performance
- Validation set metrics
- Generalization indicators

### Exporting Data

To export metrics for analysis:

1. **JSON Files**: Located in `rl_dashboard_runs/`
   - Contains all raw data
   - Easy to parse and process

2. **Manual Export**: Save Streamlit charts
   - Click the camera icon in chart
   - Saves as PNG

3. **Data Analysis**: Access JSON directly
   ```python
   import json
   with open('rl_dashboard_runs/run-YYYYMMDD-HHMMSS-xxxxxx.json') as f:
       data = json.load(f)
   ```

## Frequently Asked Questions

**Q: Can I have multiple dashboard windows open?**
A: Yes! Open the URL in multiple browser tabs/windows. Each can monitor different runs independently.

**Q: Does auto-refresh use a lot of bandwidth?**
A: No! It only checks file modification times (tiny amount of data).

**Q: Can I use this on a remote server?**
A: Yes! Port-forward with SSH:
```bash
ssh -L 8766:localhost:8766 user@server
```
Then access http://localhost:8766

**Q: What happens when training completes?**
A: Status changes to "completed", final metrics lock, you can select other runs.

**Q: Can I start a new run while monitoring an old one?**
A: Yes! New run appears in dropdown, you can switch to it anytime.

**Q: How long are runs kept?**
A: Last 40 runs are saved in `rl_dashboard_runs/`

## Support

If you encounter issues:

1. Check **DASHBOARD_AUTO_REFRESH_FIX.md** for technical details
2. Review **CODE_DIFF.md** for implementation changes
3. Check **IMPLEMENTATION_SUMMARY.md** for architecture
4. Check terminal output for error messages
5. Review browser console (F12) for JavaScript errors

---

**Happy trading! 🚀**

The dashboard is now fully capable of real-time monitoring. Happy analyzing! 📊

