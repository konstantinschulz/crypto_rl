# Data Preprocessing & Higher Time Frame (HTF) Indicators

## Overview
This directory contains scripts and documentation for data preprocessing, environment evaluation, hyperparameter optimization, and execution pipelines.

### `preprocess_htf_indicators.py`
Computes Higher Time Frame (HTF) indicators across 1-minute OHLCV crypto datasets without overwriting raw source data.

#### Motivation
With a typical microstructure observation window of `window_size = 120` to `240` minutes (2–4 hours), moving averages, z-scores, and momentum indicators only capture short-term microstructure noise. An asset may look temporarily "oversold" on a 4-hour timeframe while plunging in a macro downtrend on the daily chart.

By precomputing and embedding multi-timeframe HTF features directly into the dataset and feeding them through `precalculate_static_obs`, the RL agent gains macro awareness (trend alignment vs counter-trend divergence) with minimal additional observation dimensions.

---

## Indicators Computed

For each symbol time series (sorted chronologically):

1. **`htf_slope_15m` (15-Minute EMA Slope)**:
   $$\text{EMA}_{15}(t) = \text{EWM}(\text{close}, \text{span}=15)$$
   $$\text{htf\_slope\_15m}(t) = \frac{\text{EMA}_{15}(t) - \text{EMA}_{15}(t - 15)}{\text{close}(t) + 10^{-8}}$$
   - Captures intermediate momentum and micro-trend slope over a 15-minute timeframe.

2. **`htf_slope_1h` (1-Hour EMA Slope)**:
   $$\text{EMA}_{60}(t) = \text{EWM}(\text{close}, \text{span}=60)$$
   $$\text{htf\_slope\_1h}(t) = \frac{\text{EMA}_{60}(t) - \text{EMA}_{60}(t - 60)}{\text{close}(t) + 10^{-8}}$$
   - Captures hourly trend strength and direction.

3. **`htf_regime_24h` (24-Hour / Daily Trend Regime)**:
   $$\text{EMA}_{1440}(t) = \text{EWM}(\text{close}, \text{span}=1440)$$
   $$\text{htf\_regime\_24h}(t) = \frac{\text{close}(t) - \text{EMA}_{1440}(t)}{\text{EMA}_{1440}(t) + 10^{-8}}$$
   - Measures the relative deviation and regime position of the asset against the 24-hour daily trend (protecting the agent from buying into daily cliff drops).

---

## Output Dataset Schema

The augmented parquet file contains:

| Column | Type | Description |
|---|---|---|
| `symbol` | `dictionary<values=string>` | Trading pair (e.g. `BTCUSDT`) |
| `open_time` | `int64` | Bar open timestamp in milliseconds |
| `open_time_dt` | `timestamp[ns, UTC]` | Bar open datetime (UTC) |
| `open` | `float32` | Open price |
| `high` | `float32` | High price |
| `low` | `float32` | Low price |
| `close` | `float32` | Close price |
| `volume` | `float32` | Volume |
| `htf_slope_15m` | `float32` | 15-minute trend slope |
| `htf_slope_1h` | `float32` | 1-hour trend slope |
| `htf_regime_24h` | `float32` | 24-hour daily trend regime |

---

## Usage

```bash
./.conda/bin/python scripts/preprocess_htf_indicators.py \
    --input binance_spot_1m_last4y_single.parquet \
    --output binance_spot_1m_last4y_single_htf.parquet \
    --row-group-size 43200 \
    --compression zstd
```
