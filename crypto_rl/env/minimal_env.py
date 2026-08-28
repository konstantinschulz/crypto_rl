import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crypto_rl.config import RLConfig
from crypto_rl.env.action_processing import (
    apply_continuous_action,
    apply_discrete_action,
)
from crypto_rl.env.data_utils import pivot_ohlcv
from crypto_rl.env.feature_utils import (
    MACRO_DIM,
    STATIC_PER_ASSET_DIM,
    precalculate_static_obs,
)
from crypto_rl.env.logging_utils import flush_log_parquet, init_log, log_action
from crypto_rl.env.metrics import get_per_asset_summary
from crypto_rl.env.observation import build_observation


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
        prices_arr: np.ndarray,
        static_obs: np.ndarray,
        asset_names: list[str],
        config: RLConfig,
        run_id: str = "default",
        is_eval: bool = False,
    ):
        super().__init__()
        # Preprocessed data supplied externally
        self.prices_arr: np.ndarray = prices_arr
        self.precalc_static_obs: np.ndarray = static_obs
        self.asset_names: list[str] = asset_names
        self.num_assets: int = len(asset_names)
        # Preserve DataFrame attributes for compatibility when resetting from parquet
        self.prices_df = None
        self.open_df = None
        self.high_df = None
        self.low_df = None
        self.volume_df = None
        self.config = config
        self.window_size = config.window_size
        self.run_id = run_id
        self.fee_rate = config.fee_rate
        self.is_eval = is_eval
        self.reward_type = config.reward_type
        self.hold_cost_rate = config.hold_cost_rate
        self.empty_buy_penalty = config.empty_buy_penalty
        self.empty_sell_penalty = config.empty_sell_penalty
        self.illegal_sell_penalty = config.illegal_sell_penalty
        self.illegal_buy_penalty = config.illegal_buy_penalty
        self.drawdown_penalty_coef = config.drawdown_penalty_coef
        self.profit_bonus = config.profit_bonus
        self.min_turnover_threshold = config.min_turnover_threshold
        # Per-asset performance metrics
        self.per_asset_realized_pnl = np.zeros(self.num_assets, dtype=np.float32)
        self.per_asset_trades = np.zeros(self.num_assets, dtype=np.int32)
        self.per_asset_wins = np.zeros(self.num_assets, dtype=np.int32)
        self.per_asset_fees = np.zeros(self.num_assets, dtype=np.float32)

        self.max_single_step_allocation = config.max_single_step_allocation
        self.disable_logging = config.disable_logging
        self.fees_paid_total = 0.0
        self.previous_drawdown = 0.0
        self.last_invalid_sell = False
        self.action_dead_zone = config.action_dead_zone
        self.hold_incentive = config.hold_incentive
        self.peak_portfolio_value = config.budget_initial
        self.action_space_type = config.action_space_type
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32
            )
        elif self.action_space_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete([3, self.num_assets, 101])

        # Dimension breakdown:
        # 1. Macro features: btc_mom_norm (1), btc_vol_norm (1), btc_trend_regime 1-hot (3) -> 5
        # 2. Base window price changes: window_size * num_assets
        # 3. 1-min 30-bar price changes: 30 * num_assets
        # 4. 5-min 24-bar price changes: 24 * num_assets
        # 5. 60-min 24-bar price changes: 24 * num_assets
        # 6. Statistical indicators: 7 * num_assets (vol_norm, intrabar_vol, vwap_dev, mom_norm, rsi, macd, rel_mom)
        # 7. Portfolio features: cash_pct (1) + holdings_pct (N) + unrealised_pnl_pct (N) + has_position (N) + current_drawdown (1) = 2 + 3 * num_assets
        self.macro_dim = MACRO_DIM
        self.static_per_asset_dim = STATIC_PER_ASSET_DIM
        self.static_dim = self.macro_dim + (self.static_per_asset_dim * self.num_assets)
        self.has_position_dim = self.num_assets

        # Dynamic windows calculated on the fly in observation.py: (W + 30 + 24 + 24) * N
        dynamic_windows_dim = (config.window_size + 78) * self.num_assets
        # Total observation dimension: macro(5) + dynamic_windows((W+78)*N) + indicators(7*N) + portfolio(2 + 3*N)
        obs_dim = (
            self.macro_dim
            + dynamic_windows_dim
            + (self.static_per_asset_dim * self.num_assets)
            + 2
            + (3 * self.num_assets)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Pre-allocate a single static buffer for observations to eliminate np.concatenate
        self.obs_buf = np.zeros(obs_dim, dtype=np.float32)
        # Pre-allocate dynamic slice buffers
        self.unrealised_pnl_buf = np.zeros(self.num_assets, dtype=np.float32)

        self.current_step = self.window_size
        self.cash = config.budget_initial
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = config.budget_initial
        self.episode_count = 0

        self.log_buffer = []
        self.log_file_path = None
        self.last_remap_note = None
        self.avg_entry_price = np.zeros(self.num_assets, dtype=np.float32)
        # New counters for tracking wins
        self.winning_trades_count = 0
        self.total_closed_trades = 0
        self.total_cost_basis = np.zeros(self.num_assets, dtype=np.float32)
        self.parquet_path: str | None = config.parquet_path
        self.n_rows: int = config.n_rows

        self._cached_valid_open_times: np.ndarray | None = None
        self._cached_symbols: list[str] | None = None
        self._cached_k: int | None = None

        if self.parquet_path is not None and self.n_rows > 0:
            from crypto_rl.data import get_valid_start_timestamps

            self._cached_valid_open_times, self._cached_symbols, self._cached_k = (
                get_valid_start_timestamps(self.parquet_path, n=self.n_rows)
            )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.disable_logging and self.log_buffer:
            flush_log_parquet(self)

        self.episode_count += 1

        if self.parquet_path is not None:
            from crypto_rl.data import (
                get_valid_start_timestamps,
                read_window_from_timestamps,
            )

            if self._cached_valid_open_times is None:
                (
                    self._cached_valid_open_times,
                    self._cached_symbols,
                    self._cached_k,
                ) = get_valid_start_timestamps(self.parquet_path, n=self.n_rows)

            new_df = read_window_from_timestamps(
                self.parquet_path,
                self._cached_valid_open_times,
                self._cached_symbols,
                self._cached_k,
            )

            # Pivot the OHLCV data
            prices_piv, open_piv, high_piv, low_piv, volume_piv = pivot_ohlcv(new_df)

            # Ensure a fixed asset universe by reindexing to match original self.asset_names
            self.prices_df = (
                prices_piv.reindex(columns=self.asset_names).ffill().bfill().fillna(0.0)
            )
            self.open_df = (
                open_piv.reindex(columns=self.asset_names).ffill().bfill().fillna(0.0)
            )
            self.high_df = (
                high_piv.reindex(columns=self.asset_names).ffill().bfill().fillna(0.0)
            )
            self.low_df = (
                low_piv.reindex(columns=self.asset_names).ffill().bfill().fillna(0.0)
            )
            self.volume_df = (
                volume_piv.reindex(columns=self.asset_names).ffill().bfill().fillna(0.0)
            )

            assert self.prices_df.shape[1] == self.num_assets, (
                f"Asset count mismatch! Expected {self.num_assets} coins, "
                f"but sampled window only contained {self.prices_df.shape[1]}."
            )
            precalculate_static_obs(self)

        if not self.disable_logging:
            init_log(self, run_id=self.run_id)

        self.current_step = self.window_size
        self.cash = self.config.budget_initial
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = self.config.budget_initial
        self.fees_paid_total = 0.0
        self.trades_count = 0
        self.avg_entry_price = np.zeros(self.num_assets, dtype=np.float32)
        self.total_cost_basis = np.zeros(self.num_assets, dtype=np.float32)
        # Reset new counters
        self.winning_trades_count = 0
        self.total_closed_trades = 0
        self.peak_portfolio_value = self.config.budget_initial
        self.per_asset_realized_pnl.fill(0.0)
        self.per_asset_trades.fill(0)
        self.per_asset_wins.fill(0)
        self.per_asset_fees.fill(0.0)
        self.previous_drawdown = 0.0
        return self._get_obs(), {"fees_paid": 0.0}

    def close(self) -> None:
        """Cleanly close resources and flush pending logs."""
        if not self.disable_logging and self.log_buffer:
            flush_log_parquet(self)
        super().close()

    def _get_obs(self) -> np.ndarray:
        """Delegate observation construction to the observation module."""
        return build_observation(self)

    def step(self, action):
        """Refactored step method delegating action handling to helper functions."""
        prev_portfolio_value = self.portfolio_value
        current_prices = self.prices_arr[self.current_step - 1]
        next_prices = self.prices_arr[self.current_step]

        fee_paid = 0.0
        # Retrieve any step penalty set by discrete action processing
        step_penalty = getattr(self, "_step_penalty", 0.0)
        realised_pnl = 0.0

        is_valid_sell = False
        self.last_remap_note = None
        # --- SNAPSHOT HOLDINGS BEFORE ACTION ---
        old_holdings = np.copy(self.holdings)
        if self.action_space_type == "continuous":
            # Snapshot old asset exposure fraction before rebalancing (for logging)
            _pre_asset_value = np.sum(self.holdings * current_prices)
            _pre_port = self.cash + _pre_asset_value
            _old_asset_frac = (
                (_pre_asset_value / _pre_port) if _pre_port > 1e-8 else 0.0
            )
            # Continuous action processing moved to helper
            fee_paid, trade_units = apply_continuous_action(self, action)
        elif self.action_space_type == "multidiscrete":
            # Discrete action processing moved to helper
            fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price = (
                apply_discrete_action(self, action)
            )
            asset_idx = action[1]
            if fee_paid > 0:
                self.per_asset_fees[asset_idx] += fee_paid
        # 1. Initialize a decomposition tracker
        reward_components = {
            "market_alpha": 0.0,
            "hold_cost": 0.0,
            "profit_bonus": 0.0,
            "drawdown_penalty": 0.0,
            "rule_penalties": -step_penalty,  # From invalid buys/sells
            "hold_incentive": 0.0,
            "terminal_return": 0.0,
        }
        # --- CALCULATE PER-ASSET MULTI-TRADE METRICS ---
        deltas = self.holdings - old_holdings
        for i, delta in enumerate(deltas):
            # 1. BOUGHT (Scaled in)
            if delta > 1e-8:
                # Value of existing units AT INITIAL COST BASIS
                value_of_existing = old_holdings[i] * self.avg_entry_price[i]

                # Cost of newly acquired units (Delta is the number of units)
                cost_of_new = delta * current_prices[i]
                cost_of_new_with_fees = cost_of_new * (1.0 + self.fee_rate)

                # Update weighted average entry price
                self.avg_entry_price[i] = (
                    value_of_existing + cost_of_new_with_fees
                ) / self.holdings[i]

            # 2. SOLD (Scaled out)
            elif delta < -1e-8:
                self.total_closed_trades += 1
                self.per_asset_trades[i] += 1
                # SAFEGUARD: Clamp amount sold to what we actually owned to kill phantom PnL
                amount_sold = min(abs(delta), old_holdings[i])
                # Real revenue generated minus fees
                revenue = amount_sold * current_prices[i] * (1.0 - self.fee_rate)
                # Cost basis of what was just sold
                cost_basis = amount_sold * self.avg_entry_price[i]
                # Instead of a flat bonus, scale it by the profit margin
                trade_margin = (
                    (revenue - cost_basis) / cost_basis if cost_basis > 1e-8 else 0.0
                )
                allocation = (
                    cost_basis / prev_portfolio_value
                    if prev_portfolio_value > 1e-8
                    else 0.0
                )
                # The net equity contribution of this specific trade to the total portfolio
                adjusted_pnl = trade_margin * allocation * prev_portfolio_value
                self.per_asset_realized_pnl[i] += adjusted_pnl
                # A win requires net revenue to exceed the exact cost we paid for those units
                if revenue > cost_basis:
                    self.winning_trades_count += 1
                    self.per_asset_wins[i] += 1
                    reward_components["profit_bonus"] += (
                        self.profit_bonus * trade_margin
                    )
                # If we fully closed out, reset cost basis to 0
                if self.holdings[i] < 1e-8:
                    self.avg_entry_price[i] = 0.0
                    # SAFEGUARD: Snap negative holdings to 0 to kill the short-selling exploit
                    self.holdings[i] = 0.0
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.prices_arr.shape[0]
        current_asset_value = np.sum(self.holdings * next_prices)
        self.portfolio_value = self.cash + current_asset_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        # Range: 0.0 (at peak) down to -1.0 (-100% loss)
        current_drawdown = (
            self.portfolio_value - self.peak_portfolio_value
        ) / self.peak_portfolio_value
        # Calculate delta. If current (-0.10) is worse than previous (-0.05), delta is -0.05.
        # We use min(0, ...) to ensure we ONLY capture worsening drawdowns, ignoring recoveries.
        delta_drawdown = min(0.0, current_drawdown - self.previous_drawdown)
        # Update previous drawdown for the next step
        self.previous_drawdown = current_drawdown
        # Reward calculation
        portfolio_return = (
            (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            if prev_portfolio_value > 0
            else 0.0
        )
        asset_returns = np.divide(
            next_prices - current_prices,
            current_prices,
            out=np.zeros_like(current_prices),
            where=current_prices > 1e-8,
        )
        market_return = np.mean(asset_returns)

        if self.reward_type == "excess_return":
            alpha_diff = portfolio_return - market_return
            # Slightly penalize underperformance, but avoid the 200x asymmetric distortion
            if alpha_diff < 0:
                alpha_diff *= 1.2
            reward_components["market_alpha"] = alpha_diff
            reward_components["drawdown_penalty"] = (
                delta_drawdown * self.drawdown_penalty_coef
            )
            # Scale hold cost by the initial budget to keep it relative
            if not done and current_asset_value > 0:
                reward_components["hold_cost"] = (
                    -(current_asset_value / self.config.budget_initial)
                    * self.hold_cost_rate
                )
        else:
            reward_components["market_alpha"] = (
                self.portfolio_value - prev_portfolio_value
            )
        # --- ACTION-SPECIFIC INCENTIVES (Optional) ---
        if self.action_space_type == "continuous":
            n_held = sum(
                1
                for i in range(self.num_assets)
                if abs(action[i]) <= self.action_dead_zone
            )
            reward_components["hold_incentive"] = n_held * self.hold_incentive
        info: dict[str, Any] = {}
        if done:
            terminal_return = (
                self.portfolio_value - self.config.budget_initial
            ) / self.config.budget_initial
            reward_components["terminal_return"] = terminal_return
            # Capture the final state right before the auto-reset
            info["per_asset_stats"] = get_per_asset_summary(self)
            info["final_portfolio_value"] = self.portfolio_value
            info["final_trades_count"] = self.trades_count
            info["final_fees_paid"] = self.fees_paid_total

        if step_penalty >= 0.1:
            logging.debug(f"High step penalty value (>= 0.1): {step_penalty}")

        # Explicitly clip raw rewards at the source to prevent variance explosion
        reward = np.clip(sum(reward_components.values()), -1.0, 1.0)
        if not self.disable_logging:
            if self.action_space_type == "continuous":
                exp_act = np.exp(action - np.max(action))
                weights = exp_act / np.sum(exp_act)
                # Determine net direction of this rebalance for logging purposes.
                # Compare new asset-exposure fraction to the pre-trade snapshot taken
                # at the top of step().  Positive delta → buying assets (BUY=1),
                # negative delta → reducing assets (SELL=2), flat → HOLD=0.
                _new_asset_value = np.sum(self.holdings * next_prices)
                _new_port = self.cash + _new_asset_value
                _new_asset_frac = (
                    _new_asset_value / _new_port if _new_port > 1e-8 else 0.0
                )
                _delta_frac = _new_asset_frac - _old_asset_frac
                _turnover_threshold = 1e-4
                if _delta_frac > _turnover_threshold:
                    _log_action_type = 1  # BUY
                elif _delta_frac < -_turnover_threshold:
                    _log_action_type = 2  # SELL
                else:
                    _log_action_type = 0  # HOLD
                eff_action = np.array(
                    [_log_action_type, np.argmax(weights[1:]), weights[0] * 100.0]
                )
                log_action(
                    self,
                    self.current_step,
                    eff_action,
                    reward,
                    prev_portfolio_value,
                    fee=fee_paid,
                    reward_components=reward_components,
                )
            elif self.action_space_type == "multidiscrete":
                # Discrete action logging remains unchanged (action variables are updated inside helper)
                log_action(
                    self,
                    self.current_step,
                    action,
                    reward,
                    prev_portfolio_value,
                    trade_price=trade_price,
                    trade_units=trade_units,
                    fee=fee_paid,
                    reward_components=reward_components,
                )
        info |= {
            "fees_paid": self.fees_paid_total,
            "trades_count": self.trades_count,
            "total_closed_trades": self.total_closed_trades,
            "winning_trades_count": self.winning_trades_count,
            "realised_pnl": realised_pnl,
            "is_valid_sell": is_valid_sell,
            "episode_count": self.episode_count,
            "reward_components": reward_components,
        }
        return (
            self._get_obs(),
            float(reward),
            done,
            False,
            info,
        )

    def render(self):
        pass
