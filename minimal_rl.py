import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import os

# Try to use pyarrow for efficient, row-group-aware parquet reads so we
# don't load the entire file into memory. Fallback to pandas.read_parquet
# if pyarrow is not available.
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

BUDGET_INITIAL = 100.0

class MinimalCryptoEnv(gym.Env):
    """
    The absolute simplest Crypto Trading Environment.
    - Observation: Last 10 price changes.
    - Action: 0 (Stay in Cash) or 1 (Invest in Crypto).
    - Reward: The literal change in Portfolio Value ($ PnL) from the action.
    """
    def __init__(self, prices, window_size=10):
        super().__init__()
        self.prices = prices
        self.window_size = window_size

        # Action is 0 (Cash) or 1 (Crypto)
        self.action_space = spaces.Discrete(2)
        # Observation is the last 10 relative price changes
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = BUDGET_INITIAL
        return self._get_obs(), {}

    def _get_obs(self):
        # Return the last 'window_size' prices as simple percentage changes
        window = self.prices[self.current_step - self.window_size : self.current_step]
        normalized = (window / window[-1]) - 1.0
        return normalized.astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step - 1]
        next_price = self.prices[self.current_step]

        # Advance time
        self.current_step += 1
        done = self.current_step >= len(self.prices)

        if done:
            return self._get_obs(), 0.0, done, False, {}

        # Calculate PnL based on the action
        if action == 1:
            # We are holding crypto: Portfolio changes based on price movement
            price_change_pct = (next_price - current_price) / current_price
            pnl = self.portfolio_value * price_change_pct
        else:
            # We are holding cash: No change
            pnl = 0.0

        self.portfolio_value += pnl

        # Simply reward the direct PnL we just made
        reward = pnl

        return self._get_obs(), float(reward), done, False, {}

def main():
    parser = argparse.ArgumentParser(description="Minimal RL training with optional dashboard reporting")
    parser.add_argument("--rows", type=int, default=10000, help="Number of last rows to load from parquet")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps to train")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard state updates")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to write run state (if dashboard enabled)")
    args = parser.parse_args()

    print("1. Loading raw data...")
    # Load only the last N minutes of price data to avoid OOM.
    # We prefer a row-group-aware reader (pyarrow) so only required
    # row-groups are loaded. If pyarrow isn't available, fall back to
    # a pandas read of only the 'close' column and then tail().
    def read_last_n_close(path, n=10000):
        if pq is None:
            # Fallback: read only the close column (may still be large)
            df_local = pd.read_parquet(path, columns=["close"])
            return df_local['close'].dropna().values[-n:]

        # Use pyarrow. Parquet files are divided into row groups; find
        # which row groups contain the last `n` rows and read only them.
        pf = pq.ParquetFile(path)
        rg_counts = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        total_rows = sum(rg_counts)
        start_row = max(0, total_rows - n)

        # Find first row_group that intersects start_row
        cum = 0
        first_rg = 0
        for i, cnt in enumerate(rg_counts):
            if cum + cnt > start_row:
                first_rg = i
                break
            cum += cnt

        # Read from first_rg to the last row group
        rgs = list(range(first_rg, pf.num_row_groups))
        table = pf.read_row_groups(rgs, columns=["close"])  # pyarrow.Table
        df_local = table.to_pandas()

        # It's possible we read a bit more than needed; take the tail.
        return df_local['close'].dropna().values[-n:]

    # Load last 10k minutes (~7 days) by default but keep it configurable
    prices = read_last_n_close("binance_spot_1m_last4y_single.parquet", n=args.rows)

    # Split 80/20 into train and test
    split_idx = int(len(prices) * 0.8)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    print("2. Setting up environment...")
    train_env = MinimalCryptoEnv(train_prices)

    # If dashboard integration is requested, prepare run directory and callback
    run_dir = None
    state_file = None
    index_file = Path("rl_dashboard_index.json")

    class DashboardCallback(BaseCallback):
        def __init__(self, state_path: Path, check_freq: int = 500):
            super().__init__()
            self.state_path = state_path
            self.check_freq = check_freq
            self.start_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        def _on_training_start(self) -> None:
            # Initialize state
            self._write_state(status="initializing")

        def _on_step(self) -> bool:
            # Periodically write state with current num_timesteps
            if self.num_timesteps % self.check_freq == 0:
                self._write_state(status="running")
            return True

        def _on_training_end(self) -> None:
            self._write_state(status="finished")

        def _write_state(self, status="running"):
            try:
                state = {
                    "run": {
                        "run_id": run_id if 'run_id' in globals() else "run-unknown",
                        "mode": "minimal",
                        "status": status,
                        "started_at": self.start_ts,
                        "finished_at": None if status != "finished" else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "current_step": int(self.num_timesteps),
                        "progress_pct": int(100 * min(1.0, float(self.num_timesteps) / float(args.timesteps))),
                    },
                    "technical": {"loss": {"train": None}},
                    "series": {
                        "train_reward": [{"step": int(self.num_timesteps), "value": None}],
                        "portfolio_value": [{"step": int(self.num_timesteps), "value": None}],
                    },
                    "finance": {},
                }
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
            except Exception:
                pass

    callback = None
    if args.dashboard:
        # Create run directory
        run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S-minimal")
        run_dir = Path(args.run_dir) if args.run_dir else Path("rl_dashboard_runs") / run_id
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
                "started_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            runs = index.get("runs", [])
            runs.insert(0, run_entry)
            index["runs"] = runs
            index["latest_model_run"] = run_entry
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass

        callback = DashboardCallback(state_file, check_freq=max(1, args.timesteps // 100 if args.timesteps >= 100 else 1))

    print("3. Training simplest PPO model...")
    model = PPO("MlpPolicy", train_env, verbose=1, seed=42)
    if callback is not None:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    else:
        model.learn(total_timesteps=args.timesteps)

    print("4. Testing the trained model...")
    test_env = MinimalCryptoEnv(test_prices)
    obs, _ = test_env.reset()
    done = False
    action_1_count = 0
    steps = 0

    while not done:
        # Predict the action using our trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)

        if action == 1:
            action_1_count += 1
        steps += 1

    print("-" * 30)
    print("RESULTS:")
    print(f"Final Test Portfolio Value: ${test_env.portfolio_value:.2f}")
    print(f"Time in Market (Crypto):    {action_1_count}/{steps} steps ({action_1_count/steps*100:.1f}%)")

    # Baseline comparison (if we just bought and held)
    buy_hold_return = (test_prices[-1] - test_prices[0]) / test_prices[0]
    buy_hold_final = 100.0 * (1 + buy_hold_return)
    print(f"Buy/Hold Baseline:          ${buy_hold_final:.2f}")

if __name__ == "__main__":
    main()

