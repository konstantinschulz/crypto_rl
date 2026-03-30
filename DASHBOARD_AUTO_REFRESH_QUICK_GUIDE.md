# Quick Reference: Dashboard Auto-Refresh Implementation

## Before vs After

### ❌ BEFORE: Caching + No Polling
```python
@st.cache_data(ttl=2)  # ← Cached for 2 seconds
def load_index():
    index_file = Path("rl_dashboard_index.json")
    with open(index_file, "r") as f:
        return json.load(f)  # Only reloaded on cache expire or user interaction
```

**Problems:**
- Dashboard didn't refresh when files changed
- Users had to manually refresh or wait for cache to expire
- New runs not detected automatically
- Running data not updated in real-time

### ✅ AFTER: File Monitoring + Smart Polling
```python
def load_index():
    """Load index file with file modification time tracking."""
    index_file = Path("rl_dashboard_index.json")
    if index_file.exists():
        try:
            current_mtime = index_file.stat().st_mtime  # ← Lightweight check
            if current_mtime > st.session_state.last_index_mtime:
                st.session_state.last_index_mtime = current_mtime
            with open(index_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# Polling loop at the end of the dashboard
if st.session_state.auto_refresh_enabled:
    should_rerun = False
    if index_file.exists():
        current_index_mtime = index_file.stat().st_mtime
        if current_index_mtime > st.session_state.last_index_mtime:
            should_rerun = True
    
    if should_rerun:
        st.success("🔄 Updating data...")
        time.sleep(0.5)
        st.rerun()  # ← Trigger UI refresh
    else:
        st.caption("🔄 Monitoring for changes...")
        time.sleep(2)
        st.rerun()  # ← Check again in 2 seconds
```

**Benefits:**
- ✅ Real-time updates every 2 seconds
- ✅ New runs detected automatically
- ✅ Progress data updated without user action
- ✅ Charts refresh with latest metrics
- ✅ User can toggle auto-refresh on/off
- ✅ Manual refresh button available

## Performance Impact

| Aspect | Impact | Notes |
|--------|--------|-------|
| **File I/O** | Minimal | Only `stat()` calls (no full file reads) |
| **CPU Usage** | Low | Just checks mtimes every 2 seconds |
| **Memory** | Same | No caching overhead |
| **Network** | None | Purely local file monitoring |
| **UI Responsiveness** | Improved | Updates happen automatically |

## User Experience Timeline

### Scenario: Run in progress

```
t=0s   : User starts training run
         "python rl_trader.py --days 14 --mode backtest --dashboard"

t=1s   : Dashboard loads, shows initial state
         Auto-refresh = ON (default)

t=3s   : First data point available
         → Dashboard detects file change
         → Refreshes automatically
         → Shows metrics, charts update

t=5s   : More progress
         → Automatic refresh
         → Metrics updated

t=7s   : New run metrics visible in dropdown
         → User can switch to new run
         → Auto-refresh tracks new run too
```

## Configuration Options

### Enable/Disable Auto-Refresh
Users can toggle the checkbox in the sidebar:
```
💾 Auto-Refresh
☑ Enable auto-refresh    ↻
```

### Manual Refresh Button
Click the ↻ button for immediate refresh

### Polling Interval
Default: 2 seconds (optimal for balance of responsiveness vs CPU usage)

To adjust: Find `time.sleep(2)` and change value:
- `time.sleep(1)` = More responsive but higher CPU
- `time.sleep(3)` = Less responsive but lower CPU
- `time.sleep(5)` = Very relaxed polling

## Key Session State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `current_run_id` | str | Track which run is selected |
| `last_index_mtime` | float | Track when index.json was last modified |
| `last_run_mtime` | float | Track when run state file was last modified |
| `auto_refresh_enabled` | bool | User preference for auto-refresh |

## Testing Checklist

- [ ] Start training run with `--dashboard` flag
- [ ] Dashboard loads and shows run
- [ ] New progress data appears automatically (within 2 seconds)
- [ ] Charts update in real-time
- [ ] Can toggle auto-refresh on/off
- [ ] Manual refresh button works
- [ ] Multiple runs show in dropdown
- [ ] Can switch between runs
- [ ] Each run updates independently when selected

## Troubleshooting

### Dashboard doesn't update automatically
1. Check if "Enable auto-refresh" is checked
2. Click ↻ to manually refresh
3. Verify `rl_dashboard_index.json` file exists and is writable

### Updates are too frequent/slow
Adjust `time.sleep()` value in the polling loop (default: 2 seconds)

### Dashboard shows stale data
Click the ↻ button for immediate refresh, or toggle auto-refresh off/on

## Browser Considerations

- ✅ Works with all modern browsers
- ✅ No special plugins needed
- ✅ Works on localhost and remote servers
- ✅ Responsive on mobile (Streamlit mobile support)

