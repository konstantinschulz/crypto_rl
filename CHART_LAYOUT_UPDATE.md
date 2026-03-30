# Dashboard Chart Layout Update

**Date**: March 30, 2026  
**Change**: Split "Portfolio vs PnL" combined chart into two separate charts  
**Status**: ✅ Complete  

---

## What Changed

### Before
```
### Portfolio vs PnL
[Combined line chart with both Portfolio Value and Realized PnL overlaid]
```

**Problem**: Both metrics on the same chart made it hard to see details, especially when values were at different scales.

### After
```
### Portfolio Value
[Separate detailed line chart for Portfolio Value]

### Realized PnL
[Separate detailed line chart for Realized PnL]
```

**Benefits**:
- ✅ Each metric has its own Y-axis scale
- ✅ Easier to see individual trends
- ✅ Better visual clarity
- ✅ More detailed analysis possible

---

## Technical Changes

**File**: `streamlit_dashboard.py` (Lines 197-215)

**Old Code**:
```python
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
```

**New Code**:
```python
# Portfolio Value
st.markdown("### Portfolio Value")
df_port = pd.DataFrame(series.get("portfolio_value", []))
if not df_port.empty and "step" in df_port.columns:
    df_port.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_port.set_index("step", inplace=True)
    st.line_chart(df_port[["value"]])

# Realized PnL
st.markdown("### Realized PnL")
df_pnl = pd.DataFrame(series.get("realized_pnl", []))
if not df_pnl.empty and "step" in df_pnl.columns:
    df_pnl.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df_pnl.set_index("step", inplace=True)
    st.line_chart(df_pnl[["value"]])
```

---

## New Dashboard Layout

The "Training Series" section now displays:

1. **Portfolio Value** - Shows wallet value over training steps
2. **Realized PnL** - Shows profit/loss from closed trades
3. **Rewards** - Training, test, and dev rewards
4. **Loss** - Training, policy, and value losses

Each chart is independent with its own scale and axis.

---

## Impact

✅ **Visual Clarity**: Much better readability  
✅ **Backward Compatible**: No breaking changes  
✅ **Same Data**: All metrics still displayed  
✅ **Better Analysis**: Easier to spot trends in each metric  
✅ **Responsive**: Works on all screen sizes  

---

## Testing

The change is automatically reflected when you refresh the dashboard. No action needed!

```bash
# Just refresh your dashboard to see the new layout
streamlit run streamlit_dashboard.py
```

The charts will now display as two separate visualizations instead of one combined chart.

