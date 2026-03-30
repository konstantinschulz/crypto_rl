# Dashboard Auto-Refresh Fix

## Problem Statement

The RL Trading Dashboard was not automatically updating when new training runs started with:
```bash
python rl_trader.py --days 14 --mode backtest --dashboard
```

Users could select runs from the dropdown, but the dashboard interface would not update automatically as new progress data became available for a specific run.

## Root Cause

The original implementation had two issues:

1. **Over-aggressive caching**: The `load_index()` function used `@st.cache_data(ttl=2)` decorator, which cached data for 2 seconds. However, Streamlit only reruns scripts when there's user interaction, not when files change on disk.

2. **No polling mechanism**: The dashboard had no way to detect when the run state files were being updated by the training process and trigger a UI refresh.

## Solution

The fix implements a **file-monitoring polling mechanism** that:

### 1. Removed Caching
- Removed the `@st.cache_data(ttl=2)` decorator from `load_index()`
- Replaced with a lightweight tracking function that monitors file modification times (mtime)

### 2. Added Auto-Refresh Session State
Three new session state variables track the dashboard state:

```python
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "last_index_mtime" not in st.session_state:
    st.session_state.last_index_mtime = 0
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
```

### 3. Implemented Smart Polling
The dashboard now:
- Checks the modification time of `rl_dashboard_index.json` and the current run's state file every 2 seconds
- Triggers a `st.rerun()` if either file has been modified since the last check
- Only updates when data actually changes (efficient)
- Shows visual feedback when updates occur

### 4. Added User Controls
A new sidebar section allows users to:
- **Enable/Disable auto-refresh** with a checkbox (default: ON)
- **Manual refresh** with a ↻ button for immediate updates
- **See status** indicating whether monitoring is active

## Code Changes

### Modified: `streamlit_dashboard.py`

#### Changes to imports:
```python
import time  # Added for polling logic
```

#### Changes to `load_index()`:
- Removed `@st.cache_data(ttl=2)` decorator
- Added file modification time tracking:
  ```python
  current_mtime = index_file.stat().st_mtime
  if current_mtime > st.session_state.last_index_mtime:
      st.session_state.last_index_mtime = current_mtime
  ```

#### Added auto-refresh UI and polling logic:
- Sidebar section with auto-refresh checkbox
- Manual refresh button
- Polling loop that checks file mtimes every 2 seconds
- Automatic rerun when changes detected

#### Simplified run selection tracking:
- Removed `load_index.clear()` call (no longer needed without cache decorator)
- Simplified to just track `current_run_id` in session state

## How It Works

1. **Initial Load**: Dashboard loads the index and current run's data normally
2. **Polling Loop**: When auto-refresh is enabled:
   - Every 2 seconds, the dashboard checks if `rl_dashboard_index.json` or the run's state file has been modified
   - File modification times are extremely lightweight to check (no file I/O)
3. **Update Detection**: When the training process writes new data, the mtime changes
4. **Automatic Refresh**: Dashboard detects the change and calls `st.rerun()` to reload data
5. **User-Friendly**: Shows visual feedback ("🔄 Updating data...") when refreshing

## Benefits

✅ **Real-time Updates**: Dashboard now updates automatically as training progresses  
✅ **Efficient**: Only checks file modification times, doesn't read entire files repeatedly  
✅ **User Control**: Can be toggled on/off via checkbox  
✅ **Manual Option**: Users can manually refresh if needed  
✅ **New Run Detection**: Automatically detects and lists new runs starting  
✅ **Visual Feedback**: Shows status indicator to confirm monitoring is active  

## Testing the Fix

To test the auto-refresh functionality:

1. **Start a training run with dashboard enabled**:
   ```bash
   python rl_trader.py --days 14 --mode backtest --dashboard
   ```

2. **Start the dashboard** (if not auto-started):
   ```bash
   streamlit run streamlit_dashboard.py
   ```

3. **Observe**:
   - New runs appear in the "Select Run" dropdown
   - Selected run's metrics update automatically
   - Charts (Portfolio, Rewards, Loss) refresh with new data
   - Progress percentage increases without user interaction

## Configuration

The polling interval (2 seconds) can be adjusted by modifying:
```python
time.sleep(2)  # Change this value to adjust polling frequency
```

Lower values = more responsive but more CPU usage  
Higher values = less responsive but lower CPU usage  

Default of 2 seconds provides a good balance.

## Backward Compatibility

✅ All existing functionality preserved  
✅ No API changes  
✅ Gracefully handles missing files  
✅ Works with or without auto-refresh enabled  

## Files Modified

- `streamlit_dashboard.py` - Added auto-refresh polling mechanism

