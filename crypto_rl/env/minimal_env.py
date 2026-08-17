import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crypto_rl.env.action_processing import (
    apply_continuous_action,
    apply_discrete_action,
)
from crypto_rl.env.data_utils import pivot_ohlcv
from crypto_rl.env.feature_utils import precalculate_static_obs
from crypto_rl.env.logging_utils import flush_log_parquet, init_log, log_action
from crypto_rl.env.observation import build_observation

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
        prices_arr: np.ndarray,
        static_obs: np.ndarray,
        asset_names: list[str],
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
        action_dead_zone: float = 0.15,
        hold_incentive: float = 0.0005,
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
        # self.num_assets and asset_names are set from provided arguments
        # self.asset_names already set from provided arguments
        self.action_dead_zone = action_dead_zone
        self.hold_incentive = hold_incentive
        self.peak_portfolio_value = BUDGET_INITIAL
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.MultiDiscrete([3, self.num_assets, 101])

        # Dimension breakdown:
        # Static: prices_flat (W*N) + vol (N) + mom (N) + rsi (N) + macd (N) = (W + 4)*N
        # Dynamic: cash_pct (1) + holdings_pct (N) + unrealised_pnl_pct (N) + has_position
        # Macro / BTC Regime features:
        # btc_mom_norm (1), btc_vol_norm (1), btc_trend_regime 1-hot: [bull, ranging, bear] (3) -> total 5 features at start of obs vector.
        self.macro_dim = 5
        self.static_per_asset_dim = (
            window_size + 7 + 78
        )  # Added multi‑timeframe price change windows: 1‑min 30, 5‑min 24, 60‑min 24
        self.static_dim = self.macro_dim + (self.static_per_asset_dim * self.num_assets)
        self.has_position_dim = self.num_assets
        obs_dim = self.static_dim + 1 + (3 * self.num_assets) + 1
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
        self.parquet_path: str | None = parquet_path
        self.n_rows: int = n_rows

        self._cached_valid_open_times: np.ndarray | None = None
        self._cached_symbols: list[str] | None = None
        self._cached_k: int | None = None

        if self.parquet_path is not None and self.n_rows > 0:
            from crypto_rl.data import get_valid_start_timestamps

            self._cached_valid_open_times, self._cached_symbols, self._cached_k = (
                get_valid_start_timestamps(self.parquet_path, n=self.n_rows)
            )

        # Static observations already precomputed; no need to recalculate here.
        # If parquet_path is provided for evaluation, static_obs will be recomputed in reset.

    # Deprecated: pivot logic moved to data_utils.pivot_ohlcv
    # def _pivot_dataframe(self, prices_df) -> pd.DataFrame:
    #     pass

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
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = BUDGET_INITIAL
        self.fees_paid_total = 0.0
        self.trades_count = 0
        self.avg_entry_price = np.zeros(self.num_assets, dtype=np.float32)
        self.total_cost_basis = np.zeros(self.num_assets, dtype=np.float32)
        self.peak_portfolio_value = BUDGET_INITIAL
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

        if self.action_space_type == "continuous":
            # Continuous action processing moved to helper
            fee_paid, trade_units = apply_continuous_action(self, action)
        else:
            # Discrete action processing moved to helper
            fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price = (
                apply_discrete_action(self, action)
            )

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

        # Reward calculation (unchanged)
        if self.reward_type == "excess_return":
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
            base_penalty = 10.0
            # ASYMMETRIC SCALING: Punish losses 2x harder than gains are rewarded
            if portfolio_return < market_return:
                alpha_diff = (portfolio_return - market_return) * (
                    base_penalty * 2
                )  # Heavier penalty for underperformance
            else:
                alpha_diff = (portfolio_return - market_return) * base_penalty
            reward_components["market_alpha"] = alpha_diff
            # DIRECT DRAWDOWN PENALTY: Continuously bleed reward the deeper the drawdown gets
            # e.g., if drawdown is -10% (-0.10), it subtracts 0.5 points per step
            drawdown_penalty_coef = 5.0
            reward_components["drawdown_penalty"] = (
                current_drawdown * drawdown_penalty_coef
            )
            if not done and current_asset_value > 0:
                reward_components["hold_cost"] = -(
                    current_asset_value * self.hold_cost_rate
                )
        else:
            reward_components["market_alpha"] = (
                self.portfolio_value - prev_portfolio_value
            )

        if self.action_space_type != "continuous":
            if realised_pnl > 0 and is_valid_sell:
                hurdle = trade_units * trade_price * 0.002
                if realised_pnl > hurdle:
                    reward_components["profit_bonus"] = self.profit_bonus
        else:
            reward_components["profit_bonus"] = self.profit_bonus
            n_held = sum(
                1
                for i in range(self.num_assets)
                if abs(action[i]) <= self.action_dead_zone
            )
            reward_components["hold_incentive"] = n_held * self.hold_incentive

        if done:
            terminal_return = (self.portfolio_value - BUDGET_INITIAL) / BUDGET_INITIAL
            terminal_reward = terminal_return * 1.0
            reward_components["terminal_return"] = terminal_reward

        if step_penalty >= 0.1:
            logging.debug(f"High step penalty value (>= 0.1): {step_penalty}")

        reward = sum(reward_components.values())
        if not self.disable_logging:
            if self.action_space_type == "continuous":
                exp_act = np.exp(action - np.max(action))
                weights = exp_act / np.sum(exp_act)
                eff_action = np.array([0, np.argmax(weights[1:]), weights[0] * 100.0])
                log_action(
                    self,
                    self.current_step,
                    eff_action,
                    reward,
                    prev_portfolio_value,
                    fee=fee_paid,
                    reward_components=reward_components,
                )
            else:
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
                "reward_components": reward_components,
            },
        )

    def render(self):
        pass
