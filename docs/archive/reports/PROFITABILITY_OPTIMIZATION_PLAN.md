# Profitability Optimization Plan

**Project**: `crypto_rl`  
**Purpose**: improve out-of-sample profitability of the RL trader after repeated negative Val/Test outcomes.

## Why the prior setup struggled

The earlier setup was likely too noisy and too reward-shaped:

- **1-minute candles** create a very noisy, fee-sensitive signal.
- The policy only saw a **single-step snapshot**, which makes trend detection difficult.
- The reward function contained several synthetic incentives that could dominate real PnL learning.
- Training on short windows makes the agent overfit to a single regime.

## First-round changes

This round focuses on the smallest coherent redesign that still preserves the current PPO workflow.

### 1) Use a coarser market timeframe

- Train on aggregated candles instead of raw 1-minute bars.
- Start with **15-minute candles**.
- Hypothesis: lower noise, fewer fee-dominated actions, better signal-to-cost ratio.

### 2) Add richer OHLCV-derived features

Use compact technical features instead of only price snapshots:

- log close
- 1-step return
- log return
- candle range pct
- volume log
- volume z-score
- EMA gap features
- MACD histogram pct
- RSI

Hypothesis: the policy can now detect momentum, exhaustion, and liquidity changes more easily.

### 3) Give the policy a rolling temporal window

- Stack the last N feature rows into the observation.
- Default starting point: **12 lookback steps**.
- Hypothesis: the agent can infer short-term trend shape without an LSTM yet.

### 4) Simplify the reward toward portfolio delta

- Use a reward based primarily on **equity / portfolio-value change**.
- Keep the reward aligned with actual net value rather than action bonuses.
- Hypothesis: less reward hacking, less churn, better transfer to Val/Test.

## Implementation scope

The first implementation pass should cover:

- `rl_trading_env.py`
  - timeframe-aware loader
  - feature engineering
  - rolling observation window
  - simplified reward path
- `rl_trader.py`
  - CLI knobs for timeframe, lookback, and reward mode
  - experiment-ready wiring of the new config values

## Initial experiment recipe

Suggested first test run:

```bash
python rl_trader.py \
  --days 14 \
  --mode backtest_3way \
  --device cpu \
  --timeframe-minutes 15 \
  --lookback-steps 12 \
  --reward-mode equity_delta \
  --reward-equity-delta-scale 100 \
  --train-steps 50000 \
  --max-symbols 3 \
  --model-arch "[256, 128, 64]"
```

## Success criteria

A run is considered promising if it shows:

- positive or near-zero **Val return**,
- positive or near-zero **Test return**,
- trade count not collapsing to zero,
- lower churn than the prior overtrading baseline,
- no obvious observation-shape or caching issues.

## Follow-up directions if this still fails

If the first-round change set does not produce a profitable model, the next candidates are:

1. **Longer history / rolling walk-forward** on months instead of days.
2. **Broader symbol universe** to improve generalization.
3. **Bigger model or recurrent policy** (e.g. LSTM / RecurrentPPO).
4. **Multi-objective reward** with a small risk penalty, not action incentives.
5. **Alternative execution constraints** such as stricter trade suppression or longer hold times.

## Notes

This file is a working optimization plan, not a final diagnosis. The goal of the first round is to establish whether the model can learn a real market signal when the input and objective are made more realistic.

