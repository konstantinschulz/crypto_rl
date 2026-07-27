"""
crypto_rl.env
=============
Gymnasium trading environment for multiple crypto assets.
"""

import json
import logging
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

BUDGET_INITIAL = 100.0


class MinimalCryptoEnv(gym.Env):
    """
    A highly optimized Crypto Trading Environment with multiple assets and variable trade amounts.

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
        parquet_path: str | None = None,
        n_rows: int = 0,
        action_space_type: str = "continuous",
        max_single_step_allocation: float = 0.5,
        disable_logging: bool = False,  # <-- Set True during Optuna HPO!
    ):
        super().__init__()
        self.prices_df = self._pivot_dataframe(prices_df)
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
        self.max_single_step_allocation = max_single_step_allocation
        self.disable_logging = disable_logging
        self.fees_paid_total = 0.0
        self.last_invalid_sell = False
        self.num_assets = self.prices_df.shape[1]
        self.asset_names = self.prices_df.columns.tolist()

        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
            )
        else:
            self.action_space = spaces.MultiDiscrete([3, self.num_assets, 101])

        # Dimension breakdown:
        # Static: prices_flat (W*N) + vol (N) + mom (N) + rsi (N) + macd (N) = (W + 4)*N
        # Dynamic: cash_pct (1) + holdings_pct (N) + unrealised_pnl_pct (N) + has_position (N) = 1 + 3*N
        self.static_per_asset_dim = window_size + 5
        self.macro_dim = 2
        self.static_dim = (self.static_per_asset_dim * self.num_assets) + self.macro_dim
        self.has_position_dim = self.num_assets
        obs_dim = self.static_dim + 1 + (3 * self.num_assets)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Pre-allocate a single static buffer for observations to eliminate np.concatenate
        self.obs_buf = np.zeros(obs_dim, dtype=np.float32)
        # Pre-allocate dynamic slice buffers
        self.unrealised_pnl_buf = np.zeros(self.num_assets, dtype=np.float32)

        self.current_step = self.window_size
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = BUDGET_INITIAL
        self.episode_count = 0

        self.log_buffer = []
        self.log_file_path = None
        self.last_remap_note = None
        self.avg_entry_price = np.zeros(self.num_assets, dtype=np.float32)
        self.total_cost_basis = np.zeros(self.num_assets, dtype=np.float32)
        self.parquet_path: str | None = None
        self.n_rows: int = 0

        self._precalculate_indicators()

    def _pivot_dataframe(self, prices_df) -> pd.DataFrame:
        # Forward-fill and backward-fill to prevent missing altcoin bars from injecting NaNs into the observation pipeline.
        return (
            prices_df.pivot(index="open_time", columns="symbol", values="close")
            .ffill()
            .bfill()
        )

    def _precalculate_indicators(self) -> None:
        """Pre-calculates and packs scale-invariant features and BTC macro signals."""
        self.prices_arr = self.prices_df.values.astype(np.float32)
        T = self.prices_df.shape[0]
        N = self.num_assets
        W = self.window_size

        # 1. Base Returns & Rolling Volatility per coin
        returns_df = self.prices_df.pct_change()
        vol_df = returns_df.rolling(window=W, min_periods=W).std()

        # 2. Rolling Momentum per coin
        momentum_df = (self.prices_df / self.prices_df.shift(W)) - 1.0

        # 3. RSI [-1, 1]
        delta = self.prices_df.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs_df = avg_gain / (avg_loss + 1e-8)
        rsi_df = (rs_df / (1 + rs_df)) * 2.0 - 1.0

        # 4. Normalized MACD
        rolling_window = self.prices_df.rolling(window=W, min_periods=W)
        mean_all = rolling_window.mean()
        std_all = rolling_window.std()
        mean_3 = self.prices_df.rolling(window=3, min_periods=3).mean()
        macd_df = (mean_3 - mean_all) / (std_all + 1e-8)

        # 5. NEW: BTC Macro Features & Relative Strength
        btc_symbol = (
            "BTCUSDT"
            if "BTCUSDT" in self.prices_df.columns
            else self.prices_df.columns[0]
        )
        btc_mom = momentum_df[btc_symbol]
        btc_vol = vol_df[btc_symbol]

        # Relative Strength: (Altcoin Return) - (BTC Return) over window W
        rel_mom_df = momentum_df.sub(btc_mom, axis=0)

        # -------------------------------------------------------------
        # Time-Series Normalization (Rolling 100-bar Z-Score)
        # Replaces broken cross-sectional (axis=1) normalization!
        # -------------------------------------------------------------
        def rolling_zscore(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            r = df.rolling(window=window, min_periods=10)
            return (df - r.mean()) / (r.std() + 1e-8)

        vol_norm = rolling_zscore(vol_df).fillna(0.0).values.astype(np.float32)
        mom_norm = rolling_zscore(momentum_df).fillna(0.0).values.astype(np.float32)
        rel_mom_norm = rolling_zscore(rel_mom_df).fillna(0.0).values.astype(np.float32)
        macd_raw = macd_df.fillna(0.0).values.astype(np.float32)
        rsi_raw = rsi_df.fillna(0.0).values.astype(np.float32)

        btc_mom_norm = (
            rolling_zscore(btc_mom.to_frame())[btc_symbol]
            .fillna(0.0)
            .values.astype(np.float32)
        )
        btc_vol_norm = (
            rolling_zscore(btc_vol.to_frame())[btc_symbol]
            .fillna(0.0)
            .values.astype(np.float32)
        )

        # PACK ALL STATIC FEATURES INTO MASTER MATRIX (T+1, static_dim)
        self.precalc_static_obs = np.zeros((T + 1, self.static_dim), dtype=np.float32)

        for t in range(W, T):
            window = self.prices_arr[t - W : t]
            last_price = self.prices_arr[t - 1]
            norm_win = ((window / last_price) - 1.0).ravel()

            idx = 0
            # Pack window relative prices
            self.precalc_static_obs[t, idx : idx + W * N] = norm_win
            idx += W * N

            # Pack per-asset normalized indicators
            self.precalc_static_obs[t, idx : idx + N] = vol_norm[t]
            idx += N
            self.precalc_static_obs[t, idx : idx + N] = mom_norm[t]
            idx += N
            self.precalc_static_obs[t, idx : idx + N] = rsi_raw[t]
            idx += N
            self.precalc_static_obs[t, idx : idx + N] = macd_raw[t]
            idx += N
            self.precalc_static_obs[t, idx : idx + N] = rel_mom_norm[t]  # NEW
            idx += N

            # Pack global BTC macro features
            self.precalc_static_obs[t, idx] = btc_mom_norm[t]  # NEW
            self.precalc_static_obs[t, idx + 1] = btc_vol_norm[t]  # NEW

        # Handle terminal boundary
        self.precalc_static_obs[T] = self.precalc_static_obs[T - 1]

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _init_log(self, run_id: str = "default") -> None:
        """Initialize per-run action log path and clear in-memory log buffer."""
        log_root = Path("logs")
        run_log_dir = log_root / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        prefix = "actions_eval" if self.is_eval else "actions"
        self.log_file_path = (
            run_log_dir / f"{prefix}_ep{self.episode_count}_{int(time.time())}.parquet"
        )
        self.log_buffer = []

    def _flush_log_parquet(self) -> None:
        """Write buffered log entries to a single Parquet file at the end of an episode."""
        if not self.disable_logging and self.log_buffer and self.log_file_path:
            try:
                df_log = pd.DataFrame(self.log_buffer)
                df_log.to_parquet(self.log_file_path, index=False)
            except Exception as e:
                print(
                    f"Warning: Failed to write action log to {self.log_file_path}: {e}"
                )
            self.log_buffer = []

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
        """Record step details into the in-memory log buffer."""
        if not self.disable_logging:
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
                "episode": int(self.episode_count),
                "step": int(step),
                "action_type": action_type_str,
                "symbol": asset_str,
                "amount_pct": float(action[2]),
                "reward": float(reward),
                "portfolio": float(portfolio),
                "note": self.last_remap_note if self.last_remap_note else "",
                "price": float(trade_price),
                "units": float(trade_units),
                "fee": float(fee),
            }

            self.log_buffer.append(entry)
            self.last_invalid_sell = False
            self.last_remap_note = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Flush previous episode logs to Parquet before starting a new episode
        if not self.disable_logging and self.log_buffer:
            self._flush_log_parquet()

        self.episode_count += 1

        # Tier 3A: resample a fresh random time window each episode when enabled.
        if self.parquet_path is not None:
            from crypto_rl.data import read_last_n

            new_df = read_last_n(self.parquet_path, n=self.n_rows)
            self.prices_df = self._pivot_dataframe(new_df)
            assert self.prices_df.shape[1] == self.num_assets, (
                f"Asset count mismatch! Expected {self.num_assets} coins, "
                f"but sampled window only contained {self.prices_df.shape[1]}."
            )
            # Re-run precalculation on the newly selected price distribution
            self._precalculate_indicators()

        if not self.disable_logging:
            self._init_log(run_id=self.run_id)

        self.current_step = self.window_size
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = BUDGET_INITIAL
        self.fees_paid_total = 0.0
        self.trades_count = 0
        self.avg_entry_price = np.zeros(self.num_assets, dtype=np.float32)
        self.total_cost_basis = np.zeros(self.num_assets, dtype=np.float32)
        return self._get_obs(), {"fees_paid": 0.0}

    def close(self) -> None:
        """Cleanly close resources and flush pending logs."""
        if not self.disable_logging and self.log_buffer:
            self._flush_log_parquet()
        super().close()

    def _get_obs(self) -> np.ndarray:
        # 1. Zero-copy write of all static features (prices_flat, vol, mom, rsi, macd)
        self.obs_buf[: self.static_dim] = self.precalc_static_obs[self.current_step]

        # 2. Fast scalar calculations
        current_prices = self.prices_arr[self.current_step - 1]
        asset_value = np.sum(self.holdings * current_prices)
        total_val = self.cash + asset_value

        idx = self.static_dim
        if total_val > 0:
            self.obs_buf[idx] = self.cash / total_val
            idx += 1
            # Write holdings_pct directly into the buffer slice
            self.obs_buf[idx : idx + self.num_assets] = (
                self.holdings * current_prices
            ) / total_val
        else:
            self.obs_buf[idx] = 0.0
            idx += 1
            self.obs_buf[idx : idx + self.num_assets] = 0.0

        idx += self.num_assets

        # 3. Fast boolean masking for unrealised PnL (Bypasses slow np.divide ufunc)
        self.unrealised_pnl_buf.fill(0.0)
        mask = self.avg_entry_price > 0
        if np.any(mask):
            self.unrealised_pnl_buf[mask] = (
                current_prices[mask] - self.avg_entry_price[mask]
            ) / self.avg_entry_price[mask]

        self.obs_buf[idx : idx + self.num_assets] = self.unrealised_pnl_buf
        idx += self.num_assets

        # 4. Binary per-asset position flags
        has_position = (self.holdings > 1e-9).astype(np.float32)
        self.obs_buf[idx : idx + self.num_assets] = has_position

        return self.obs_buf

    def step(self, action):
        prev_portfolio_value = self.portfolio_value

        # Pull step pricing boundaries from high-speed pre-allocated Numpy slices
        current_prices = self.prices_arr[self.current_step - 1]
        next_prices = self.prices_arr[self.current_step]

        trade_price = 0.0
        trade_units = 0.0
        fee_paid = 0.0
        self.last_remap_note = None
        realised_pnl = 0.0
        step_penalty = 0.0
        is_valid_sell = False
        profit_bonus_reward = 0.0

        if self.action_space_type == "continuous":
            threshold = 0.05

            # Sells: action[i] < -threshold
            sells = []
            for i in range(self.num_assets):
                act_val = action[i]
                if act_val < -threshold:
                    amount_pct = np.clip(abs(act_val), 0.0, 1.0)
                    buy_fraction = min(amount_pct, self.max_single_step_allocation)
                    sells.append((i, amount_pct))

            trade_prices = np.zeros(self.num_assets, dtype=np.float32)
            trade_units_dict = np.zeros(self.num_assets, dtype=np.float32)
            fees_paid_dict = np.zeros(self.num_assets, dtype=np.float32)

            # Execute sells
            for asset_idx, amount_pct in sells:
                if self.holdings[asset_idx] > 0:
                    units_to_sell = self.holdings[asset_idx] * amount_pct
                    if units_to_sell > 0:
                        is_valid_sell = True
                        t_price = current_prices[asset_idx]
                        t_units = units_to_sell
                        gross_proceeds = units_to_sell * t_price
                        fee_paid_asset = gross_proceeds * self.fee_rate
                        proceeds = gross_proceeds - fee_paid_asset
                        self.cash += proceeds
                        self.holdings[asset_idx] -= t_units
                        self.fees_paid_total += fee_paid_asset
                        fee_paid += fee_paid_asset
                        self.trades_count += 1

                        trade_prices[asset_idx] = t_price
                        trade_units_dict[asset_idx] = t_units
                        fees_paid_dict[asset_idx] = fee_paid_asset

                        if self.holdings[asset_idx] <= 1e-9:
                            self.avg_entry_price[asset_idx] = 0.0
                            self.total_cost_basis[asset_idx] = 0.0
                        else:
                            self.total_cost_basis[asset_idx] *= self.holdings[
                                asset_idx
                            ] / (self.holdings[asset_idx] + units_to_sell)

                        asset_realised_pnl = proceeds - (
                            t_units * self.avg_entry_price[asset_idx]
                        )
                        realised_pnl += asset_realised_pnl

                        # Profit bonus hurdle
                        if asset_realised_pnl > 0:
                            hurdle = t_units * t_price * 0.002
                            if asset_realised_pnl > hurdle:
                                profit_bonus_reward += self.profit_bonus
                else:
                    # Illegal SELL: no holdings
                    # step_penalty += self.illegal_sell_penalty * self.portfolio_value
                    pass

            # Buys: action[i] > threshold
            buys = []
            for i in range(self.num_assets):
                act_val = action[i]
                if act_val > threshold:
                    amount_pct = np.clip(act_val, 0.0, 1.0)
                    buys.append((i, amount_pct))

            sum_buys = sum(amount_pct for _, amount_pct in buys)
            cash_available = self.cash

            for asset_idx, amount_pct in buys:
                if cash_available <= 1e-9:
                    # step_penalty += self.illegal_buy_penalty * self.portfolio_value
                    continue

                # Normalize buy amount if sum of buy fractions > 1.0
                if sum_buys > 1.0:
                    buy_fraction = amount_pct / sum_buys
                else:
                    buy_fraction = amount_pct

                buy_amount_usd = cash_available * buy_fraction
                if buy_amount_usd > 0:
                    t_price = current_prices[asset_idx]
                    fee_paid_asset = buy_amount_usd * self.fee_rate
                    amount_after_fee = buy_amount_usd - fee_paid_asset
                    t_units = amount_after_fee / t_price
                    self.cash -= buy_amount_usd
                    self.holdings[asset_idx] += t_units
                    self.fees_paid_total += fee_paid_asset
                    fee_paid += fee_paid_asset
                    self.trades_count += 1
                    self.total_cost_basis[asset_idx] += amount_after_fee
                    self.avg_entry_price[asset_idx] = (
                        self.total_cost_basis[asset_idx] / self.holdings[asset_idx]
                        if self.holdings[asset_idx] > 0
                        else 0.0
                    )

                    trade_prices[asset_idx] = t_price
                    trade_units_dict[asset_idx] = t_units
                    fees_paid_dict[asset_idx] = fee_paid_asset
        else:
            # MultiDiscrete logic (original)
            action_type = action[0]
            asset_idx = action[1]
            amount_pct = float(action[2]) / 100.0
            amount_pct = np.clip(amount_pct, 0.0, 1.0)
            if action_type == 0:
                amount_pct = 0.0

            if action_type == 1:  # Buy
                if amount_pct == 0.0:
                    step_penalty += self.empty_buy_penalty * self.portfolio_value
                    action_type = 0
                    self.last_remap_note = "empty BUY remapped to HOLD"
                elif asset_idx < self.num_assets:
                    if self.cash <= 1e-9:
                        step_penalty += self.illegal_buy_penalty * self.portfolio_value
                        action_type = 0
                        self.last_remap_note = f"illegal action (BUY, {self.asset_names[asset_idx]}, {amount_pct * 100:.0f}%): no cash, remapped to HOLD"
                        amount_pct = 0.0
                    else:
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
                                self.total_cost_basis[asset_idx]
                                / self.holdings[asset_idx]
                                if self.holdings[asset_idx] > 0
                                else 0.0
                            )
            elif action_type == 2:  # Sell
                if amount_pct == 0.0:
                    step_penalty += self.empty_sell_penalty * self.portfolio_value
                    action_type = 0
                    self.last_remap_note = "empty SELL remapped to HOLD"
                elif asset_idx < self.num_assets:
                    if self.holdings[asset_idx] > 0:
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
                        step_penalty += self.illegal_sell_penalty * self.portfolio_value
                        action_type = 0
                        self.last_remap_note = f"illegal action (SELL, {self.asset_names[asset_idx]}, {amount_pct * 100:.0f}%) remapped to HOLD"
                        amount_pct = 0.0
                else:
                    action_type = 0
                    amount_pct = 0.0
                    self.last_remap_note = "invalid asset index remapped to HOLD"

        # Advance time steps
        self.current_step += 1
        done = self.current_step >= self.prices_arr.shape[0]

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
            reward = (portfolio_return - market_return) * 10.0  # 100.0

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
        if self.action_space_type != "continuous":
            if realised_pnl > 0 and is_valid_sell:
                hurdle = trade_units * trade_price * 0.002  # 0.2 % of trade notional
                if realised_pnl > hurdle:
                    reward += self.profit_bonus  # flat — not scaled by PnL magnitude
        else:
            reward += profit_bonus_reward

        # 1B: terminal portfolio return bonus — aligns policy with terminal wealth.
        if done:
            terminal_return = (self.portfolio_value - BUDGET_INITIAL) / BUDGET_INITIAL
            reward += terminal_return * 5.0  # 10.0

        if step_penalty >= 0.1:
            logging.debug(f"High step penalty value (>= 0.1): {step_penalty}")

        # Subtract the accumulated logic penalties directly from the reward
        reward -= step_penalty

        # Log action(s)
        if not self.disable_logging:
            if self.action_space_type == "continuous":
                logged_any = False
                for i in range(self.num_assets):
                    act_val = action[i]
                    if act_val > threshold:
                        eff_action = np.array([1, i, act_val * 100.0])
                        self.log_action(
                            self.current_step,
                            eff_action,
                            reward,
                            prev_portfolio_value,
                            trade_price=trade_prices[i],
                            trade_units=trade_units_dict[i],
                            fee=fees_paid_dict[i],
                        )
                        logged_any = True
                    elif act_val < -threshold:
                        eff_action = np.array([2, i, abs(act_val) * 100.0])
                        self.log_action(
                            self.current_step,
                            eff_action,
                            reward,
                            prev_portfolio_value,
                            trade_price=trade_prices[i],
                            trade_units=trade_units_dict[i],
                            fee=fees_paid_dict[i],
                        )
                        logged_any = True

                if not logged_any:
                    eff_action = np.array([0, 0, 0.0])
                    self.log_action(
                        self.current_step,
                        eff_action,
                        reward,
                        prev_portfolio_value,
                    )
            else:
                effective_action = np.array(
                    [action_type, asset_idx, amount_pct * 100.0]
                )
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
