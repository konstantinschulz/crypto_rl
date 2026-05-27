# Trading Fee Empirical Analysis & RL Modeling Proposal

This document presents a detailed empirical analysis of the trading fees paid in your real trade history (`docs/Spot-Spot Trade History-2026-04-15-2026-05-27.xlsx`) and proposes concrete options for modeling these fees in the virtual RL environment (`minimal_rl.py`).

---

## 1. Empirical Analysis of Trade History

We analyzed **342 trades** spanning **2026-04-15 to 2026-05-27** across **26 trading pairs** (primarily `ONDO_USDT`, `ORDI_USDT`, `DYDX_USDT`, and `STRK_USDT`).

### Key Findings

1. **Fee Currency**: 
   All fees (100%) were settled in **USDT** (the quote asset), even for BUY trades. This simplifies fee modeling because all cash additions and deductions occur in the same currency (`USDT`/`cash`).

2. **Maker vs. Taker Fees**:
   The trade history contains both Maker and Taker roles:
   * **Maker Trades**: 48 trades (14%)
   * **Taker Trades**: 294 trades (86%)

   | Role | Trade Count | Fee Rate (Fee / Total) | Notes |
   | :--- | :---: | :---: | :--- |
   | **Maker** | 48 | **0.00%** (0.0000) | Always 0% across all assets and dates |
   | **Taker** | 294 | **0.05%** (0.0005) or **0.00%** | 255 trades paid exactly 0.05%; 39 trades paid 0% |

3. **The Zero-Fee Taker Mystery**:
   All **39 zero-fee Taker trades** were on a single asset: **`ONDO_USDT`**.
   * **Before 2026-05-12**: ONDO_USDT taker trades were charged the standard **0.05%** fee.
   * **On/After 2026-05-22**: ONDO_USDT taker trades paid **0%** fees.
   This perfectly reflects a real-world **Zero-Fee Spot Trading Promotion** (commonly offered by exchanges like Bybit, OKX, or Binance for promotional pairs or stablecoins) that took effect around mid-May.

---

## 2. Options for Virtual RL Environment Modeling

When modeling fees in a virtual RL environment, we must balance **realism**, **agent behavior**, and **robustness**.

### Option A: Flat Taker Fee (0.05% / 0.0005) - *Recommended*
Since the RL backtest environment executes trades instantly at 1-minute close prices (acting as market orders / liquidity taking), we model all transactions with a flat **0.05%** fee.

* **For BUY**:
  $$\text{qty} = \frac{\text{cash} \times \text{amount\_pct} \times (1 - \text{fee\_rate})}{\text{price}}$$
  $$\text{cash} \leftarrow \text{cash} - (\text{cash} \times \text{amount\_pct})$$
* **For SELL**:
  $$\text{gross\_proceeds} = \text{qty\_to\_sell} \times \text{price}$$
  $$\text{proceeds} = \text{gross\_proceeds} \times (1 - \text{fee\_rate})$$
  $$\text{cash} \leftarrow \text{cash} + \text{proceeds}$$
* **Pros**: Simple, highly robust, matches the empirical average taker rate, and prevents cash from ever dropping below zero.

### Option B: Maker-Taker Role Probability
We simulate that a portion of the agent's trades are filled as Maker (0.0% fee) and others as Taker (0.05% fee) using a stochastic split (e.g. 85% Taker, 15% Maker) or dynamic logic.
* **Pros**: Captures the exact fee variance from your spreadsheet.
* **Cons**: Historically executing at close prices is 100% Taker. Simulating Maker fill without simulating order book depth/limit orders is inaccurate and can make the agent overly optimistic.

### Option C: Conservative Fee + Slippage Proxy (0.07% to 0.10%)
We model a flat fee slightly higher than the actual 0.05% fee (e.g., 0.07% or 0.10%) to act as a **slippage and spread proxy**.
* **Pros**: Extremely important for RL agents. Without a slippage buffer, agents will learn to over-trade to capture tiny, unrealistic 0.01% price moves that are impossible to execute in real life due to bid-ask spread and order book depth.

---

## 3. Concrete Implementation Plan in `minimal_rl.py`

We propose implementing **Option A with an adjustable parameter (Option C ready)** in `MinimalCryptoEnv` of `minimal_rl.py`.

### Code Changes to `MinimalCryptoEnv`

```python
# 1. In __init__, add a fee_rate parameter:
def __init__(self, prices_df: pd.DataFrame, window_size=10, run_id: str = 'default', fee_rate: float = 0.0005):
    ...
    self.fee_rate = fee_rate
    self.fees_paid_total = 0.0

# 2. In reset, reset cumulative fees:
def reset(self, seed=None, options=None):
    ...
    self.fees_paid_total = 0.0
    return self._get_obs(), {}

# 3. In step, update execution and track fees:
def step(self, action):
    ...
    trade_price = 0.0
    trade_units = 0.0
    fee_paid = 0.0
    
    if action_type == 1:  # Buy
        if self.cash > 0 and asset_idx < self.num_assets:
            buy_amount_usd = self.cash * amount_pct
            if buy_amount_usd > 0:
                trade_price = current_prices[asset_idx]
                fee_paid = buy_amount_usd * self.fee_rate
                amount_after_fee = buy_amount_usd - fee_paid
                trade_units = amount_after_fee / trade_price
                
                self.cash -= buy_amount_usd
                self.holdings[asset_idx] += trade_units
                self.fees_paid_total += fee_paid
                
    elif action_type == 2:  # Sell
        if self.holdings[asset_idx] > 0 and asset_idx < self.num_assets:
            units_to_sell = self.holdings[asset_idx] * amount_pct
            if units_to_sell > 0:
                trade_price = current_prices[asset_idx]
                trade_units = units_to_sell
                gross_proceeds = units_to_sell * trade_price
                fee_paid = gross_proceeds * self.fee_rate
                proceeds = gross_proceeds - fee_paid
                
                self.cash += proceeds
                self.holdings[asset_idx] -= trade_units
                self.fees_paid_total += fee_paid
```

### Action Log and CLI Support
We can also:
1. Update `log_action` to log `fee_paid`.
2. Add a `--fee-rate` command-line argument in `main()` with a default of `0.0005` (0.05%).
3. Return `fees_paid` in the environment's `info` dictionary and include it in the dashboard state updates.
