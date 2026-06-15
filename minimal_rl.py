import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
# Import PPO lazily to avoid heavy dependency during unit tests
try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore
# Import BaseCallback lazily; define a no‑op fallback if unavailable
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover
    class BaseCallback:  # type: ignore
        """Fallback BaseCallback with minimal interface used in this script."""
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            # Return a dummy callable for any attribute used in the code
            return lambda *a, **k: None

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

# Try to use pyarrow for efficient, row-group-aware parquet reads so we
# don't load the entire file into memory. Fallback to pandas.read_parquet
# if pyarrow is not available.
try:
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
except Exception:
    pq = None
    ds = None

BUDGET_INITIAL = 100.0

class MinimalCryptoEnv(gym.Env):
    """
    A slightly less minimal Crypto Trading Environment with multiple assets and variable trade amounts.
    - Observation: Last `window_size` price changes for all assets.
    - Action: MultiDiscrete space [3, num_assets, 101]:
        - Index 0: action_type: 0 (Hold), 1 (Buy), 2 (Sell)
        - Index 1: asset_idx: Index of the asset to act on
        - Index 2: amount_pct: Percentage of cash/holdings (0 to 100, mapped to 0.0-1.0)
    - Reward: The literal change in Portfolio Value ($ PnL).
    """
    def __init__(self, prices_df: pd.DataFrame, window_size=10, run_id: str = 'default', fee_rate: float = 0.0007, reward_type: str = 'excess_return', is_eval: bool = False):
        super().__init__()
        # Pivot the dataframe so columns are symbols, index is timestamp, values are 'close'
        self.prices_df = prices_df.pivot(index='open_time', columns='symbol', values='close')
        self.window_size = window_size
        self.run_id = run_id
        self.fee_rate = fee_rate
        self.is_eval = is_eval
        self.reward_type = reward_type
        self.fees_paid_total = 0.0
        self.last_invalid_sell = False
        self.num_assets = self.prices_df.shape[1]
        self.asset_names = self.prices_df.columns.tolist()

        # Action is MultiDiscrete:
        # [action_type (3), asset_idx (num_assets), amount_pct (101)]
        self.action_space = spaces.MultiDiscrete([3, self.num_assets, 101])

        # Observation is the last 'window_size' relative price changes for all assets,
        # plus the cash percentage, plus the holdings percentage for all assets.
        obs_dim = (window_size * self.num_assets) + 1 + self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self.current_step = self.window_size
        self.cash = BUDGET_INITIAL
        self.holdings = np.zeros(self.num_assets) # units of each asset
        self.portfolio_value = BUDGET_INITIAL
        self.episode_count = 0
        
        self.action_log = None
        self.log_file_path = None

    def _init_log(self, run_id: str = 'default'):
        """Initialize per-run action log."""
        # Create a dedicated subdirectory for this run (e.g., logs/run-20260602-130000-minimal)
        log_root = Path('logs')
        run_log_dir = log_root / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        # Log file now only needs episode and timestamp info
        prefix = "actions_eval" if self.is_eval else "actions"
        self.log_file_path = run_log_dir / f"{prefix}_ep{self.episode_count}_{int(time.time())}.jsonl"
        self.action_log = open(self.log_file_path, 'a', encoding='utf-8')

    def log_action(self, step: int, action: np.ndarray, reward: float, portfolio: float, trade_price: float = 0.0, trade_units: float = 0.0, fee: float = 0.0):
        """Record step details in a storage-efficient, human-readable JSONL format."""
        if self.action_log:
            action_type_idx = int(action[0])
            action_types = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_type_str = action_types.get(action_type_idx, 'UNKNOWN')
            
            asset_idx = int(action[1])
            asset_str = self.asset_names[asset_idx] if 0 <= asset_idx < len(self.asset_names) else 'UNKNOWN'
            
            entry = {
                'step': step,
                'action_type': action_type_str,
                'symbol': asset_str,
                'amount_pct': float(action[2]),
                'reward': float(reward),
                'portfolio': float(portfolio),
            }
            if self.last_remap_note:
                entry['note'] = self.last_remap_note
            if trade_price > 0:
                entry['price'] = float(trade_price)
                entry['units'] = float(trade_units)
            if fee > 0:
                entry['fee'] = float(fee)
            
            self.action_log.write(json.dumps(entry) + '\n')
            if step % 1000 == 0:
                self.action_log.flush()
            # Reset flag after logging
            self.last_invalid_sell = False
            self.last_remap_note = None

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
        return self._get_obs(), {'fees_paid': 0.0}

    def close(self):
        """Cleanly close resources."""
        if self.action_log:
            self.action_log.close()
            self.action_log = None
        super().close()

    def _get_obs(self):
        # Return the last 'window_size' prices as simple percentage changes for all assets
        window = self.prices_df.iloc[self.current_step - self.window_size : self.current_step]
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
            
        # Concatenate prices and allocation
        obs = np.concatenate([
            prices_flat,
            np.array([cash_pct], dtype=np.float32),
            holdings_pct.astype(np.float32)
        ])
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
        if action_type == 1:  # Buy
            if amount_pct == 0.0:
                # *** Empty BUY: 0% amount ***
                # Penalise the agent by deducting a small fraction of its cash.
                penalty = 0.001 * self.cash
                self.cash -= penalty
                action_type = 0
                self.last_remap_note = "empty BUY remapped to HOLD"
            # Ensure we have cash and the asset index is valid
            elif self.cash > 0 and asset_idx < self.num_assets:
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
        elif action_type == 2:  # Sell
            if amount_pct == 0.0:
                # *** Empty SELL: 0% amount ***
                # Penalise the agent by deducting a small fraction of its cash.
                penalty = 0.001 * self.cash
                self.cash -= penalty
                action_type = 0
                self.last_remap_note = "empty SELL remapped to HOLD"
            # Validate asset index first
            elif asset_idx < self.num_assets:
                if self.holdings[asset_idx] > 0:
                    # Normal sell path (holdings exist)
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
                        self.trades_count += 1
                else:
                    # *** Illegal SELL: no holdings ***
                    penalty = 0.005 * self.cash  # use high penalty for quick learning
                    self.cash -= penalty
                    action_type = 0
                    self.last_remap_note = f"illegal action (SELL, {self.asset_names[asset_idx]}, {amount_pct*100:.0f}%) remapped to HOLD"
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

        if done:
            # On done, liquidate all holdings to get final portfolio value
            final_asset_value = np.sum(self.holdings * next_prices)
            self.portfolio_value = self.cash + final_asset_value
            reward = self.portfolio_value - prev_portfolio_value # Reward for the last step
            # Reward calculation for done step
            if self.reward_type == 'excess_return':
                portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
                asset_returns = (next_prices - current_prices) / current_prices
                market_return = np.mean(asset_returns)
                reward = (portfolio_return - market_return) * 100.0
            else:
                reward = self.portfolio_value - prev_portfolio_value
            
            # Log the effective amount used
            effective_action = np.array([action_type, asset_idx, amount_pct * 100.0])
            self.log_action(self.current_step, effective_action, reward, prev_portfolio_value, trade_price, trade_units, fee_paid)
            return self._get_obs(), float(reward), done, False, {'fees_paid': self.fees_paid_total, 'trades': self.trades_count}

        # Update portfolio value based on new prices for held assets
        current_asset_value = np.sum(self.holdings * next_prices)
        self.portfolio_value = self.cash + current_asset_value

        # Reward calculation for normal step
        if self.reward_type == 'excess_return':
            portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
            asset_returns = (next_prices - current_prices) / current_prices
            market_return = np.mean(asset_returns)
            reward = (portfolio_return - market_return) * 100.0
        else:
            reward = self.portfolio_value - prev_portfolio_value
        
        # Log the effective amount used
        effective_action = np.array([action_type, asset_idx, amount_pct * 100.0])
        self.log_action(self.current_step, effective_action, reward, prev_portfolio_value, trade_price, trade_units, fee_paid)

        return self._get_obs(), float(reward), done, False, {'fees_paid': self.fees_paid_total, 'trades': self.trades_count}

def main():
    parser = argparse.ArgumentParser(
        description="Minimal RL training with optional dashboard reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minimal_rl.py --rows 10000 --timesteps 20000
  python minimal_rl.py --dashboard --rows 15000 --timesteps 50000 --run-dir rl_dashboard_runs
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="subset.parquet",
        help="Path to Parquet dataset file (default: subset.parquet)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of last rows to load from parquet file (default: 10000 ~7 days). "
             "Lower values use less RAM; higher values gives more training data. "
             "Uses pyarrow row-group-aware reading if available for memory efficiency."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="Total timesteps to train PPO model (default: 20000)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable dashboard integration: creates run entry in rl_dashboard_index.json "
             "and writes periodic state.json for live monitoring in streamlit_dashboard.py"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory where run state will be written (if --dashboard is used). "
             "Default: rl_dashboard_runs/"
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Trading fee rate (flat percentage of trade volume, e.g. 0.001 for 0.1%)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Observation window size in minutes (default: 10)"
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Seed for data subset selection (default: 42)"
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="excess_return",
        choices=["pnl", "excess_return"],
        help="Reward function type to use for training (default: excess_return)"
    )
    args = parser.parse_args()

    # Apply data seed for reproducible dataset selection
    np.random.seed(args.data_seed)

    print("1. Loading raw data...")
    # Load only the last N minutes of price data to avoid OOM.
    # We prefer a row-group-aware reader (pyarrow) so only required
    # row-groups are loaded. If pyarrow isn't available, fall back to
    # a pandas read of only the 'close' column and then tail().
    def read_last_n(path, n=10000):
        # Read a random contiguous time window for a set of major symbols
        cols = ['symbol', 'open_time', 'close']
        if ds is None or pq is None:
            # Fallback path if pyarrow is unavailable
            df_local = pd.read_parquet(path, columns=cols)
            df_local = df_local.dropna()
            available_symbols = list(df_local['symbol'].unique())
            symbols = [s for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'] if s in available_symbols]
            if not symbols:
                symbols = available_symbols[:5]
            num_assets = len(symbols)
            k = max(1, n // num_assets)
            
            df_sub = df_local[df_local['symbol'].isin(symbols)]
            btc_times = df_sub[df_sub['symbol'] == symbols[0]]['open_time'].unique()
            btc_times.sort()
            if len(btc_times) <= k:
                t_start = btc_times[0]
                t_end = btc_times[-1]
            else:
                start_idx = np.random.randint(0, len(btc_times) - k)
                t_start = btc_times[start_idx]
                t_end = btc_times[start_idx + k - 1]
            df_local = df_sub[(df_sub['open_time'] >= t_start) & (df_sub['open_time'] <= t_end)]
            return df_local.sort_values(by=['open_time', 'symbol']).reset_index(drop=True)

        dataset = ds.dataset(path, format='parquet')
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
        num_assets = len(symbols)
        k = max(1, n // num_assets)

        # Load open times of BTCUSDT to determine a random time window
        btc_filter = ds.field('symbol') == 'BTCUSDT'
        btc_open_times = dataset.to_table(columns=['open_time'], filter=btc_filter).column('open_time').to_numpy()

        if len(btc_open_times) <= k:
            t_start = btc_open_times[0]
            t_end = btc_open_times[-1]
        else:
            start_idx = np.random.randint(0, len(btc_open_times) - k)
            t_start = btc_open_times[start_idx]
            t_end = btc_open_times[start_idx + k - 1]

        # Read the aligned data for the chosen symbols during this time window
        query_filter = ds.field('symbol').isin(symbols) & (ds.field('open_time') >= t_start) & (ds.field('open_time') <= t_end)
        table = dataset.to_table(columns=cols, filter=query_filter)
        df_local = table.to_pandas()
        return df_local.dropna().sort_values(by=['open_time', 'symbol']).reset_index(drop=True)

    # Load data with required columns
    prices_df = read_last_n(args.dataset, n=args.rows)

    # Split 80/20 into train and test
    split_idx = int(len(prices_df) * 0.8)
    train_prices_df = prices_df.iloc[:split_idx]
    test_prices_df = prices_df.iloc[split_idx:]

    # If dashboard integration is requested, prepare run directory and callback
    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")
    
    print("2. Setting up environment...")
    train_env = MinimalCryptoEnv(train_prices_df, window_size=args.window_size, run_id=run_id, fee_rate=args.fee_rate, reward_type=args.reward_type)

    run_dir = None
    state_file = None
    index_file = Path("rl_dashboard_index.json")

    class DashboardCallback(BaseCallback):
        def __init__(self, state_path: Path, window_size: int, reward_type: str, check_freq: int = 500):
            super().__init__()
            self.state_path = state_path
            self.window_size = window_size
            self.reward_type = reward_type
            self.check_freq = check_freq
            self.start_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.last_portfolio_value = BUDGET_INITIAL
            
            # Series data
            self.series = {
                "train_reward": [],
                "portfolio_value": [],
                "trades": [],
                "win_rate": [],
                "train_loss": [],
                "policy_loss": [],
                "value_loss": [],
                "approx_kl": [],
                "clip_fraction": [],
                "ram_mb": [],
                "total_return_pct": [],
                "drawdown_pct": [],
            }

            self.current_trades = 0
            self.winning_trades = 0
            self.peak_portfolio_value = BUDGET_INITIAL
            
            try:
                import psutil
                self.psutil = psutil
            except ImportError:
                self.psutil = None

        def _on_training_start(self) -> None:
            self._write_state(status="initializing")

        def _on_step(self) -> bool:
            # Periodically write state with current num_timesteps and real metrics
            if self.num_timesteps % self.check_freq == 0:
                self._collect_metrics()
                self._write_state(status="running")
            return True

        def _on_training_end(self) -> None:
            self._collect_metrics()
            self._write_state(status="finished")

        def _collect_metrics(self):
            """Extract real portfolio value and reward from the training environment."""
            step = int(self.num_timesteps)
            try:
                # Use ep_info_buffer for smooth finalized episode metrics
                mean_ep_rew = 0.0
                if len(self.model.ep_info_buffer) > 0:
                    mean_ep_rew = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer]))
                    current_portfolio = BUDGET_INITIAL + mean_ep_rew
                else:
                    # Fallback to current step if no episode finished yet
                    portfolio_values = self.training_env.get_attr('portfolio_value')
                    current_portfolio = float(portfolio_values[0]) if portfolio_values else BUDGET_INITIAL

                holdings_list = self.training_env.get_attr('holdings')
                current_holdings = holdings_list[0] if holdings_list else np.zeros(1)

                # Simplified trade count
                if np.sum(np.abs(current_holdings)) > 1e-8:
                    self.current_trades += 1

                # Win rate placeholder: assume a "win" if mean_ep_rew > 0
                if mean_ep_rew > 0:
                    self.winning_trades += 1
                
                win_rate = (self.winning_trades / max(1, self.current_trades)) * 100.0

                # Return and Drawdown
                total_return = (current_portfolio / BUDGET_INITIAL - 1.0) * 100.0
                self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio)
                drawdown = (1.0 - current_portfolio / self.peak_portfolio_value) * 100.0

                # Append to series
                self.series["portfolio_value"].append({"step": step, "value": current_portfolio})
                self.series["train_reward"].append({"step": step, "value": mean_ep_rew})
                self.series["trades"].append({"step": step, "value": int(self.current_trades)})
                self.series["win_rate"].append({"step": step, "value": float(win_rate)})
                self.series["total_return_pct"].append({"step": step, "value": float(total_return)})
                self.series["drawdown_pct"].append({"step": step, "value": float(drawdown)})

                # Technical metrics from SB3 logger
                # SB3 uses '/' as separator, e.g., 'train/loss'
                logger_map = self.model.logger.name_to_value
                self.series["train_loss"].append({"step": step, "value": float(logger_map.get("train/loss", 0.0))})
                self.series["policy_loss"].append({"step": step, "value": float(logger_map.get("train/policy_gradient_loss", 0.0))})
                self.series["value_loss"].append({"step": step, "value": float(logger_map.get("train/value_loss", 0.0))})
                self.series["approx_kl"].append({"step": step, "value": float(logger_map.get("train/approx_kl", 0.0))})
                self.series["clip_fraction"].append({"step": step, "value": float(logger_map.get("train/clip_fraction", 0.0))})

                # Memory usage
                if self.psutil:
                    ram = self.psutil.Process().memory_info().rss / (1024 * 1024)
                    self.series["ram_mb"].append({"step": step, "value": float(ram)})

                self.last_portfolio_value = current_portfolio
            except Exception as e:
                # print(f"Error collecting metrics: {e}") # Debugging
                pass

        def _write_state(self, status="running"):
            try:
                # Keep only the last 100 entries to avoid unbounded JSON growth
                series_data = {}
                for key, data in self.series.items():
                    series_data[key] = data[-100:]

                state = {
                    "run": {
                        "run_id": run_id if 'run_id' in globals() else "run-unknown",
                        "mode": "minimal",
                        "status": status,
                        "started_at": self.start_ts,
                        "finished_at": None if status != "finished" else datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "current_step": int(self.num_timesteps),
                        "progress_pct": int(100 * min(1.0, float(self.num_timesteps) / float(args.timesteps))),
                    },
                    "technical": {
                        "loss": {"train": self.series["train_loss"][-1]["value"] if self.series["train_loss"] else None},
                        "num_data_rows": args.rows,
                        "window_size": self.window_size,
                        "reward_type": self.reward_type
                    },
                    "series": series_data,
                    "finance": {},
                }
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
            except Exception:
                pass

    callback = None
    if args.dashboard:
        # Create run directory
        run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")
        base_run_dir = Path(args.run_dir) if args.run_dir else Path("rl_dashboard_runs")
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"

        # Update index file to include this run
        try:
            index = {"runs": []}
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
            run_entry = {
                "run_id": run_id,
                "state_file": str(state_file),
                "mode": "minimal",
                "status": "initializing",
                "started_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            runs = index.get("runs", [])
            runs.insert(0, run_entry)
            index["runs"] = runs
            index["latest_model_run"] = run_entry
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass

        callback = DashboardCallback(state_file, window_size=args.window_size, reward_type=args.reward_type, check_freq=max(1, args.timesteps // 100 if args.timesteps >= 100 else 1))

    print("3. Training simplest PPO model...")
    model = PPO("MlpPolicy", train_env, verbose=1, seed=42)
    if callback is not None:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    else:
        model.learn(total_timesteps=args.timesteps)

    print("4. Testing the trained model...")
    test_env = MinimalCryptoEnv(test_prices_df, window_size=args.window_size, run_id=run_id, fee_rate=args.fee_rate, reward_type=args.reward_type, is_eval=True)  # Use same run_id for evaluation logs
    obs, _ = test_env.reset()
    done = False
    eval_trades_count = 0
    eval_winning_trades = 0
    eval_steps = 0
    eval_initial_portfolio_value = test_env.portfolio_value

    eval_portfolio_values = [{"step": 0, "value": float(test_env.portfolio_value)}]
    eval_realized_pnl = [{"step": 0, "value": 0.0}]

    while not done:
        # Predict the action using our trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)

        # Track trades for evaluation
        # Track winning steps (rough approximation of win rate)
        if reward > 0 and (action[0] == 1 or action[0] == 2):
            eval_winning_trades += 1
        eval_steps += 1

        eval_portfolio_values.append({"step": eval_steps, "value": float(test_env.portfolio_value)})
        eval_realized_pnl.append({"step": eval_steps, "value": float(test_env.portfolio_value - eval_initial_portfolio_value)})
        
    eval_trades_count = test_env.trades_count
    
    test_env.close()
    train_env.close()

    print("-" * 30)
    print("RESULTS:")
    print(f"Final Test Portfolio Value: ${test_env.portfolio_value:.2f}")
    print(f"Total Trades (Eval):        {eval_trades_count}")
    print(f"Total Fees Paid (Eval):     ${test_env.fees_paid_total:.4f}")
    eval_win_rate_pct = (eval_winning_trades / eval_trades_count * 100.0) if eval_trades_count > 0 else 0.0
    print(f"Win Rate (Eval):            {eval_win_rate_pct:.1f}%")

    # Baseline comparison (if we just bought and held the first asset)
    # Use the pivoted prices_df which is now indexed by time and contains floats.
    pivoted_df = train_env.prices_df
    first_asset = pivoted_df.columns[0]
    buy_hold_return = (pivoted_df.iloc[-1][first_asset] - pivoted_df.iloc[0][first_asset]) / pivoted_df.iloc[0][first_asset]
    buy_hold_final = BUDGET_INITIAL * (1 + buy_hold_return)
    print(f"Buy/Hold Baseline ({first_asset}): ${buy_hold_final:.2f}")

    # --- Dashboard Update for Evaluation Results ---
    if args.dashboard and state_file and state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            state["run"]["status"] = "evaluated"
            state["run"]["finished_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            state["finance"]["evaluation_results"] = {
                "final_portfolio_value": float(test_env.portfolio_value),
                "pnl": float(test_env.portfolio_value - eval_initial_portfolio_value),
                "evaluation_steps": int(eval_steps),
                "eval_trades": int(eval_trades_count),
                "eval_win_rate_pct": float(eval_win_rate_pct),
                "buy_hold_baseline": float(buy_hold_final),
                "total_fees_paid": float(test_env.fees_paid_total),
            }
            if "series" not in state:
                state["series"] = {}
            state["series"]["test_portfolio_value"] = eval_portfolio_values
            state["series"]["test_realized_pnl"] = eval_realized_pnl

            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            print(f"Dashboard state updated with evaluation results in {state_file}")
        except Exception as e:
            print(f"Error updating dashboard state with evaluation results: {e}")

if __name__ == "__main__":
    main()