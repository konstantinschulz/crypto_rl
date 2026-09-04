import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crypto_rl.config import RLConfig
from crypto_rl.env.feature_utils import (
    MACRO_DIM,
    STATIC_PER_ASSET_DIM,
)
from crypto_rl.env.logging_utils import flush_log_parquet
from crypto_rl.env.observation import build_observation
from crypto_rl.env.reset import reset_env
from crypto_rl.env.step import step_env


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
        norm_vol_arr: np.ndarray,
        asset_names: list[str],
        config: RLConfig,
        run_id: str = "default",
        is_eval: bool = False,
    ):
        super().__init__()
        # Preprocessed data supplied externally
        self.prices_arr: np.ndarray = prices_arr
        self.precalc_static_obs: np.ndarray = static_obs
        self.norm_vol_arr: np.ndarray = norm_vol_arr
        self.asset_names: list[str] = asset_names
        self.num_assets: int = len(asset_names)
        self.prices_df = None
        self.open_df = None
        self.high_df = None
        self.low_df = None
        self.volume_df = None
        self.htf_slope_15m_df = None
        self.htf_slope_1h_df = None
        self.htf_regime_24h_df = None
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
        self.max_asset_allocation = config.max_asset_allocation
        self.drawdown_penalty_coef = config.drawdown_penalty_coef
        self.profit_bonus = config.profit_bonus
        self.min_turnover_threshold = config.min_turnover_threshold
        self.target_volatility = config.target_volatility
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
        # --- NEW: Track entry steps for holding period calculations ---
        self.entry_step = np.zeros(self.num_assets, dtype=np.int32)
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
        return reset_env(self, seed=seed, options=options)

    def close(self) -> None:
        """Cleanly close resources and flush pending logs."""
        if not self.disable_logging and self.log_buffer:
            flush_log_parquet(self)
        super().close()

    def _get_obs(self) -> np.ndarray:
        """Delegate observation construction to the observation module."""
        return build_observation(self)

    def step(self, action):
        """Delegate step execution and reward logic to the step module."""
        return step_env(self, action)

    def render(self):
        pass

