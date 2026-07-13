"""
crypto_rl.env
=============
Gymnasium trading environment for multiple crypto assets.
"""

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

BUDGET_INITIAL = 100.0


class MinimalCryptoEnv(gym.Env):
    """
    A slightly less minimal Crypto Trading Environment with multiple assets and variable trade amounts.

    Observation space
    -----------------
    A flat vector composed of:
    - Last ``window_size`` relative price changes for every asset
      (normalised so the most-recent bar = 0).
    - Cash fraction of total portfolio value.
    - Per-asset holdings fraction of total portfolio value.
    - Per-asset unrealised PnL % relative to average entry price
      (0 when no position is held).

    Action space
    ------------
    ``MultiDiscrete([3, num_assets, 101])``

    - Index 0 – ``action_type``: 0 = Hold, 1 = Buy, 2 = Sell
    - Index 1 – ``asset_idx``: which asset to act on
    - Index 2 – ``amount_pct``: 0–100, mapped to 0.0–1.0 fraction of
      available cash (Buy) or holdings (Sell)

    Reward
    ------
    Controlled by ``reward_type``:

    - ``"pnl"``           – raw $ change in portfolio value per step.
    - ``"excess_return"`` – portfolio return minus equal-weight market
      return, scaled by 100 (percentage-point alpha).  A configurable
      holding-cost term and a profitable-sell bonus are also applied.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        window_size: int = 10,
        run_id: str = "default",
        fee_rate: float = 0.0007,
        reward_type: str = "excess_return",
        is_eval: bool = False,
        hold_cost_rate: float = 0.0001,
        empty_buy_penalty: float = 0.001,
        empty_sell_penalty: float = 0.001,
        illegal_sell_penalty: float = 0.005,
        illegal_buy_penalty: float = 0.005,
        trade_freq_incentive: float = 0.01,
        profit_bonus: float = 0.15,
    ):
        super().__init__()
        # Pivot the dataframe so columns are symbols, index is timestamp, values are 'close'
        self.prices_df = prices_df.pivot(
            index="open_time", columns="symbol", values="close"
        )
        self.window_size = window_size
        self.run_id = run_id
        self.fee_rate = fee_rate
        self.is_eval = is_eval
        self.reward_type = reward_type
        self.hold_cost_rate = hold_cost_rate
        self.empty_buy_penalty = empty_buy_penalty
        self.empty_sell_penalty = empty_sell_penalty
        self.illegal_sell_penalty = illegal_sell_penalty
        self.illegal_buy_penalty = illegal_buy_penalty
        self.trade_freq_incentive = trade_freq_incentive
        self.profit_bonus = profit_bonus
        self.fees_paid_total = 0.0
        self.last_invalid_sell = False
        self.num_assets = self.prices_df.shape[1]
        self.asset_names = self.prices_df.columns.tolist()

        # Action is MultiDiscrete:
        # [action_type (3), asset_idx (num_assets), amount_pct (101)]
        self.action_space = spaces.MultiDiscrete([3, self.num_assets, 101])

        # Observation: window price changes + cash_pct + holdings_pct + unrealised_pnl_pct
        obs_dim = (
            (window_size * self.num_assets) + 1 + self.num_assets + self.num_assets
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self.current_step = self.window_size
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets)  # units of each asset
        self.portfolio_value = BUDGET_INITIAL
        self.episode_count = 0

        self.action_log = None
        self.log_file_path = None
        # Internal state for tracking average entry price and total cost basis
        self.avg_entry_price = np.zeros(self.num_assets)
        self.total_cost_basis = np.zeros(self.num_assets)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _init_log(self, run_id: str = "default") -> None:
        """Initialize per-run action log."""
        # Create a dedicated subdirectory for this run (e.g., logs/run-20260602-130000-minimal)
        log_root = Path("logs")
        run_log_dir = log_root / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        # Log file now only needs episode and timestamp info
        prefix = "actions_eval" if self.is_eval else "actions"
        self.log_file_path = (
            run_log_dir / f"{prefix}_ep{self.episode_count}_{int(time.time())}.jsonl"
        )
        self.action_log = open(self.log_file_path, "a", encoding="utf-8")

    def log_action(
        self,
        step: int,
        action: np.ndarray,
        reward: float,
        portfolio: float,
        trade_price: float = 0.0,
        trade_units: float = 0.0,
        fee: float = 0.0,
    ) -> None:
        """Record step details in a storage-efficient, human-readable JSONL format."""
        if self.action_log:
            action_type_idx = int(action[0])
            action_types = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action_type_str = action_types.get(action_type_idx, "UNKNOWN")

            asset_idx = int(action[1])
            asset_str = (
                self.asset_names[asset_idx]
                if 0 <= asset_idx < len(self.asset_names)
                else "UNKNOWN"
            )

            entry = {
                "episode": self.episode_count,
                "step": step,
                "action_type": action_type_str,
                "symbol": asset_str,
                "amount_pct": float(action[2]),
                "reward": float(reward),
                "portfolio": float(portfolio),
            }
            if self.last_remap_note:
                entry["note"] = self.last_remap_note
            if trade_price > 0:
                entry["price"] = float(trade_price)
                entry["units"] = float(trade_units)
            if fee > 0:
                entry["fee"] = float(fee)

            self.action_log.write(json.dumps(entry) + "\n")
            if step % 1000 == 0:
                self.action_log.flush()
            # Reset flag after logging
            self.last_invalid_sell = False
            self.last_remap_note = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1

        # Close old log file if open to start a new file for the new episode
        if self.action_log is not None:
            self.action_log.close()
            self.action_log = None

        if self.action_log is None:
            self._init_log(run_id=self.run_id)

        self.current_step = self.window_size
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets)
        self.portfolio_value = BUDGET_INITIAL
        self.fees_paid_total = 0.0
        self.trades_count = 0
        self.avg_entry_price = np.zeros(self.num_assets)
        self.total_cost_basis = np.zeros(self.num_assets)
        return self._get_obs(), {"fees_paid": 0.0}

    def close(self) -> None:
        """Cleanly close resources."""
        if self.action_log:
            self.action_log.close()
            self.action_log = None
        super().close()

    def _get_obs(self) -> np.ndarray:
        # Return the last 'window_size' prices as simple percentage changes for all assets
        window = self.prices_df.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        # Normalize each asset's window by its last price in the window
        normalized = (window / window.iloc[-1]) - 1.0
        prices_flat = normalized.values.astype(np.float32).flatten()

        # Calculate current asset values and portfolio value
        current_prices = self.prices_df.iloc[self.current_step - 1].values
        asset_value = np.sum(self.holdings * current_prices)
        total_val = self.cash + asset_value

        if total_val > 0:
            cash_pct = self.cash / total_val
            holdings_pct = (self.holdings * current_prices) / total_val
        else:
            cash_pct = 0.0
            holdings_pct = np.zeros(self.num_assets)

        # Per-asset unrealised PnL % relative to average entry price.
        # Zero when no position is held (avg_entry_price == 0).
        unrealised_pnl_pct = np.divide(
            current_prices - self.avg_entry_price,
            self.avg_entry_price,
            out=np.zeros_like(current_prices),
            where=self.avg_entry_price != 0,
        ).astype(np.float32)

        obs = np.concatenate(
            [prices_flat, [cash_pct], holdings_pct, unrealised_pnl_pct]
        )
        return obs

    def step(self, action):
        # action is now a list/array: [action_type, asset_idx, amount_pct_int]
        action_type = action[0]
        asset_idx = action[1]
        amount_pct = float(action[2]) / 100.0
        # Ensure amount_pct is within valid range [0, 1]
        amount_pct = np.clip(amount_pct, 0.0, 1.0)
        if action_type == 0:
            amount_pct = 0.0
        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value
        # Get current and next prices for all assets
        current_prices = self.prices_df.iloc[self.current_step - 1].values
        next_prices = self.prices_df.iloc[self.current_step].values
        # Execute trade based on action
        trade_price = 0.0
        trade_units = 0.0
        fee_paid = 0.0
        self.last_remap_note = None  # Reset any previous note for this step
        realised_pnl = 0.0
        # Initialize a separate penalty tracker for this specific step
        step_penalty = 0.0
        is_valid_sell = False
        if action_type == 1:  # Buy
            if amount_pct == 0.0:
                # *** Empty BUY: 0% amount ***
                # Penalise the agent by deducting a small fraction of its cash.
                step_penalty += self.empty_buy_penalty * self.portfolio_value
                action_type = 0
                self.last_remap_note = "empty BUY remapped to HOLD"
            # Ensure the asset index is valid
            elif asset_idx < self.num_assets:
                # Ensure we have cash; use a small value slightly larger than zero to avoid rounding errors
                if self.cash <= 1e-9:
                    # *** Illegal BUY: no cash ***
                    step_penalty += self.illegal_buy_penalty * self.portfolio_value
                    action_type = 0
                    self.last_remap_note = f"illegal action (BUY, {self.asset_names[asset_idx]}, {amount_pct * 100:.0f}%): no cash, remapped to HOLD"
                    amount_pct = 0.0
                else:  # we have enough cash
                    buy_amount_usd = self.cash * amount_pct
                    if buy_amount_usd > 0:
                        trade_price = current_prices[asset_idx]
                        fee_paid = buy_amount_usd * self.fee_rate
                        amount_after_fee = buy_amount_usd - fee_paid
                        trade_units = amount_after_fee / trade_price
                        self.cash -= buy_amount_usd
                        self.holdings[asset_idx] += trade_units
                        self.fees_paid_total += fee_paid
                        self.trades_count += 1
                        self.total_cost_basis[asset_idx] += amount_after_fee
                        self.avg_entry_price[asset_idx] = (
                            self.total_cost_basis[asset_idx] / self.holdings[asset_idx]
                            if self.holdings[asset_idx] > 0
                            else 0.0
                        )
        elif action_type == 2:  # Sell
            if amount_pct == 0.0:
                # *** Empty SELL: 0% amount ***
                # Penalise the agent by deducting a small fraction of its cash.
                step_penalty += self.empty_sell_penalty * self.portfolio_value
                action_type = 0
                self.last_remap_note = "empty SELL remapped to HOLD"
            # Validate asset index first
            elif asset_idx < self.num_assets:
                if self.holdings[asset_idx] > 0:
                    # Normal sell path (holdings exist)
                    units_to_sell = self.holdings[asset_idx] * amount_pct
                    if units_to_sell > 0:
                        is_valid_sell = True
                        trade_price = current_prices[asset_idx]
                        trade_units = units_to_sell
                        gross_proceeds = units_to_sell * trade_price
                        fee_paid = gross_proceeds * self.fee_rate
                        proceeds = gross_proceeds - fee_paid
                        self.cash += proceeds
                        self.holdings[asset_idx] -= trade_units
                        self.fees_paid_total += fee_paid
                        self.trades_count += 1
                        if self.holdings[asset_idx] <= 1e-9:
                            self.avg_entry_price[asset_idx] = 0.0
                            self.total_cost_basis[asset_idx] = 0.0
                        else:
                            self.total_cost_basis[asset_idx] *= self.holdings[
                                asset_idx
                            ] / (self.holdings[asset_idx] + units_to_sell)
                        realised_pnl = proceeds - (
                            trade_units * self.avg_entry_price[asset_idx]
                        )
                else:
                    # *** Illegal SELL: no holdings ***
                    step_penalty += self.illegal_sell_penalty * self.portfolio_value
                    action_type = 0
                    self.last_remap_note = f"illegal action (SELL, {self.asset_names[asset_idx]}, {amount_pct * 100:.0f}%) remapped to HOLD"
                    amount_pct = 0.0
            else:
                # Invalid asset index – also treat as HOLD.
                action_type = 0
                amount_pct = 0.0
                self.last_remap_note = "invalid asset index remapped to HOLD"
        # else: action_type == 0 (Hold), do nothing

        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.prices_df)

        # Update portfolio value based on new prices for held assets
        current_asset_value = np.sum(self.holdings * next_prices)
        self.portfolio_value = self.cash + current_asset_value

        # Calculate base reward
        if self.reward_type == "excess_return":
            portfolio_return = (
                (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
                if prev_portfolio_value > 0
                else 0.0
            )
            asset_returns = (next_prices - current_prices) / current_prices
            market_return = np.mean(asset_returns)
            reward = (portfolio_return - market_return) * 100.0

            # Penalise open positions by a small daily carry cost (skip if done)
            if not done and current_asset_value > 0:
                hold_cost = (
                    current_asset_value * self.hold_cost_rate
                )  # configurable hold cost per step
                reward -= hold_cost
        else:
            reward = self.portfolio_value - prev_portfolio_value

        # ---------------------------------------------------------
        # IMMEDIATE REWARD SHAPING
        # Apply bonus for profitable sells exactly on the step it happens
        # ---------------------------------------------------------
        if realised_pnl > 0:
            reward += realised_pnl * self.profit_bonus  # configurable profit bonus for profitable sells
            # Add trade‑frequency incentive only during training (not eval)
            if not self.is_eval:
                # Scale incentive by the proportion of capital used in the trade
                reward += self.trade_freq_incentive * amount_pct
        # Subtract the accumulated logic penalties directly from the reward
        reward -= step_penalty
        # Log the effective amount used
        effective_action = np.array([action_type, asset_idx, amount_pct * 100.0])
        self.log_action(
            self.current_step,
            effective_action,
            reward,
            prev_portfolio_value,
            trade_price,
            trade_units,
            fee_paid,
        )
        return (
            self._get_obs(),
            float(reward),
            done,
            False,
            {
                "fees_paid": self.fees_paid_total,
                "trades_count": self.trades_count,
                "realised_pnl": realised_pnl,
                "is_valid_sell": is_valid_sell,
                "episode_count": self.episode_count,
            },
        )
