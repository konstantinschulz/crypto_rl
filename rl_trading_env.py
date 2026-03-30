"""Minimal RL trading environment with memory-efficient parquet loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from gymnasium import Env, spaces


@dataclass
class TradingConfig:
    """Trading environment configuration - optimized for memory efficiency"""
    initial_cash: float = 100.0  # Smaller starting capital = smaller portfolio tracking
    max_positions: int = 3  # Reduced from 5 for memory efficiency
    max_budget_per_trade: float = 20.0  # Smaller per-trade limit
    min_position_size: float = 5.0  # Reduced from 50
    position_duration_limit: int = 1440  # Max hold time in minutes (1 day)
    
    # Reward modeling
    profit_bonus_scale: float = 100.0  # Reward for unrealized gains
    diversification_bonus: float = 100.0  # Bonus for holding multiple coins
    drawdown_penalty: float = 50.0  # Penalty for portfolio loss
    transaction_cost: float = 0.001  # 0.1% transaction fee
    
    # Risk management
    stop_loss_pct: float = 0.05  # Emergency exit at 5% loss
    max_drawdown_pct: float = 0.20  # Max allowed drawdown from peak
    keep_history: bool = False  # NEVER keep history during training
    max_history_rows: int = 100  # Minimal history if enabled


class CryptoTradingEnv(Env):
    """Gym-compatible trading environment for cryptocurrency portfolios"""
    
    def __init__(self, df: pd.DataFrame, config: Optional[TradingConfig] = None):
        """
        Args:
            df: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
            config: TradingConfig with parameters
        """
        super().__init__()
        self.config = config or TradingConfig()

        required_cols = {'symbol', 'open_time', 'close'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        self.symbols = sorted(df['symbol'].unique().tolist())
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}

        # Keep only a compact close-price matrix in memory: [time, symbol]
        close_columns: List[np.ndarray] = []
        min_steps: Optional[int] = None
        for symbol in self.symbols:
            part = df.loc[df['symbol'] == symbol, ['open_time', 'close']]
            if part.empty:
                continue
            idx = np.argsort(part['open_time'].to_numpy(dtype=np.int64))
            symbol_close = part['close'].to_numpy(dtype=np.float32)[idx]
            if symbol_close.size == 0:
                continue
            if min_steps is None:
                min_steps = int(symbol_close.size)
            else:
                min_steps = min(min_steps, int(symbol_close.size))
            close_columns.append(symbol_close)

        if not close_columns or min_steps is None or min_steps < 2:
            raise ValueError('Not enough aligned rows to build environment.')

        trimmed = [col[-min_steps:] for col in close_columns]
        self.price_matrix = np.column_stack(trimmed).astype(np.float32)
        self.n_steps, self.n_symbols = self.price_matrix.shape
        
        # Trading state
        self.current_step = 0
        self.positions = {}  # {symbol: {'qty': float, 'entry_price': float, 'entry_step': int}}
        self.cash = self.config.initial_cash
        self.peak_portfolio_value = self.config.initial_cash
        self.history: List[Dict[str, float]] = []
        self.last_prices = self.price_matrix[0]
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.realized_pnl_total = 0.0
        self.fees_paid_total = 0.0
        self.stop_loss_exits = 0
        self.closed_trades: List[Dict] = []
        self.max_closed_trades_history = 1000  # Limit history to prevent memory growth
        
        # Action space: [action_type, symbol_idx, amount_pct]
        # action_type: 0=hold all, 1=buy, 2=sell
        # symbol_idx: which symbol (0-70)
        # amount_pct: % of available cash (0-1)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([2, len(self.symbols) - 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # State: [log-prices for all symbols] + [cash, num_positions, return, bias]
        self.observation_space = spaces.Box(
            low=-20.0, high=1e6, shape=(self.n_symbols + 4,), dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.positions = {}
        self.cash = self.config.initial_cash
        self.peak_portfolio_value = self.config.initial_cash
        self.history = []
        self.last_prices = self.price_matrix[0]
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.realized_pnl_total = 0.0
        self.fees_paid_total = 0.0
        self.stop_loss_exits = 0
        self.closed_trades = []
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading action and return new state + reward
        
        Args:
            action: [action_type (0-2), symbol_idx (0-70), amount_pct (0-1)]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        step_reward = 0.0
        
        # Limit to available steps in data
        if self.current_step >= self.n_steps:
            return self._get_observation(), step_reward, True, False, {}

        if self.current_step < 3:
            print(f"[DEBUG] step={self.current_step} RAW ACTION: {action}")
        
        prices = self.price_matrix[self.current_step]
        self.last_prices = prices
        
        # Parse action
        action_type = int(np.clip(action[0], 0, 2))
        symbol_idx = int(np.clip(action[1], 0, len(self.symbols) - 1))
        amount_pct = np.clip(action[2], 0, 1)
        symbol = self.symbols[symbol_idx]
        current_price = float(prices[symbol_idx])
        
        # Execute trading action
        if action_type == 1 and amount_pct > 0.1:  # BUY
            step_reward += self._execute_buy(symbol, current_price, amount_pct)
        elif action_type == 2 and amount_pct > 0.1:  # SELL
            step_reward += self._execute_sell(symbol, current_price, amount_pct)
        
        # Process ongoing positions
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            sym_idx = self.symbol_to_idx[sym]
            current_price = float(prices[sym_idx])
            
            # Emergency exit: stop loss
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            if pnl_pct < -self.config.stop_loss_pct:
                self._close_position(sym, current_price, reason='stop_loss')
                step_reward -= self.config.drawdown_penalty
            
            # Maximum hold time check
            elif self.current_step - pos['entry_step'] > self.config.position_duration_limit:
                self._close_position(sym, current_price, reason='time_limit')
        
        # Calculate reward components
        portfolio_value = self._get_portfolio_value(prices)
        
        # Profit bonus (only if profitable)
        unrealized_pnl = portfolio_value - self.config.initial_cash
        if unrealized_pnl > 0:
            step_reward += unrealized_pnl / 100.0  # Small bonus per step
        
        # Drawdown penalty
        if portfolio_value < self.peak_portfolio_value:
            drawdown_pct = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown_pct > self.config.max_drawdown_pct:
                step_reward -= self.config.drawdown_penalty
        
        # Diversification bonus
        if len(self.positions) > 1:
            step_reward += len(self.positions) * 0.5  # Bonus for each position
        
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        
        if self.config.keep_history:
            self.history.append({
                'step': float(self.current_step),
                'portfolio_value': float(portfolio_value),
                'positions': float(len(self.positions)),
                'cash': float(self.cash),
                'reward': float(step_reward),
            })
            if len(self.history) > self.config.max_history_rows:
                self.history = self.history[-self.config.max_history_rows:]
        
        done = self.current_step >= (self.n_steps - 1)
        
        # At the end of the episode, forcefully close all positions to realize final PnL
        if done:
            for sym in list(self.positions.keys()):
                sym_idx = self.symbol_to_idx[sym]
                final_price = float(prices[sym_idx])
                self._close_position(sym, final_price, reason='end_of_episode')
            portfolio_value = self._get_portfolio_value(prices)
            
        return self._get_observation(), step_reward, done, False, {
            'portfolio_value': portfolio_value,
            'trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0,
            'realized_pnl': self.realized_pnl_total,
        }
    
    def _execute_buy(self, symbol: str, price: float, pct: float) -> float:
        """Execute buy order with explicit constraints"""
        if symbol in self.positions:
            return 0  # Don't add to existing position (simplification)
        
        if len(self.positions) >= self.config.max_positions:
            return 0  # Already at max positions
        
        amount = self.cash * pct
        amount = min(amount, self.config.max_budget_per_trade)
        
        if amount < self.config.min_position_size:
            return 0  # Position too small
        
        # Account for transaction costs
        amount_after_fee = amount * (1 - self.config.transaction_cost)
        fee = amount * self.config.transaction_cost
        qty = amount_after_fee / price
        
        self.positions[symbol] = {
            'qty': qty,
            'entry_price': price,
            'entry_step': self.current_step
        }
        self.cash -= amount
        self.fees_paid_total += fee
        
        return -self.config.transaction_cost * amount  # Fee as negative reward
    
    def _execute_sell(self, symbol: str, price: float, pct: float) -> float:
        """Execute sell order"""
        if symbol not in self.positions:
            return 0
        
        # Sell portion or all
        pos = self.positions[symbol]
        qty_to_sell = pos['qty'] * pct if pct < 1.0 else pos['qty']
        
        gross_proceeds = qty_to_sell * price
        fee = gross_proceeds * self.config.transaction_cost
        proceeds = gross_proceeds - fee
        self.cash += proceeds
        self.fees_paid_total += fee
        
        pos['qty'] -= qty_to_sell
        if pos['qty'] < 1e-8:
            del self.positions[symbol]
        
        # Reward = realized P&L
        realized_pnl = proceeds - (qty_to_sell * pos['entry_price'])
        self._record_closed_trade(realized_pnl)
        self.closed_trades.append({
            'symbol': symbol,
            'entry_price': float(pos['entry_price']),
            'exit_price': float(price),
            'qty': float(qty_to_sell),
            'entry_step': int(pos['entry_step']),
            'exit_step': int(self.current_step),
            'realized_pnl': float(realized_pnl),
            'return_pct': float((price - pos['entry_price']) / pos['entry_price']),
            'reason': 'sell',
        })
        # Keep only last N trades to prevent memory bloat
        if len(self.closed_trades) > self.max_closed_trades_history:
            self.closed_trades = self.closed_trades[-self.max_closed_trades_history:]
        return realized_pnl / 100.0  # Normalize
    
    def _close_position(self, symbol: str, price: float, reason: str = 'forced') -> None:
        """Force-close a position (stop loss or max duration)"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            gross_proceeds = pos['qty'] * price
            fee = gross_proceeds * self.config.transaction_cost
            proceeds = gross_proceeds - fee
            self.cash += proceeds
            self.fees_paid_total += fee
            realized_pnl = proceeds - (pos['qty'] * pos['entry_price'])
            self._record_closed_trade(realized_pnl)
            self.closed_trades.append({
                'symbol': symbol,
                'entry_price': float(pos['entry_price']),
                'exit_price': float(price),
                'qty': float(pos['qty']),
                'entry_step': int(pos['entry_step']),
                'exit_step': int(self.current_step),
                'realized_pnl': float(realized_pnl),
                'return_pct': float((price - pos['entry_price']) / pos['entry_price']),
                'reason': reason,
            })
            # Keep only last N trades to prevent memory bloat
            if len(self.closed_trades) > self.max_closed_trades_history:
                self.closed_trades = self.closed_trades[-self.max_closed_trades_history:]
            if reason == 'stop_loss':
                self.stop_loss_exits += 1
            del self.positions[symbol]

    def _record_closed_trade(self, realized_pnl: float) -> None:
        self.total_trades += 1
        self.realized_pnl_total += float(realized_pnl)
        if realized_pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def _get_portfolio_value(self, prices: np.ndarray) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos['qty'] * float(prices[self.symbol_to_idx[sym]])
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value

    def get_metrics_snapshot(self, prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Export compact finance metrics for evaluation and dashboard reporting."""
        prices = self.last_prices if prices is None else prices
        portfolio_value = float(self._get_portfolio_value(prices))
        unrealized_pnl = portfolio_value - self.config.initial_cash - self.realized_pnl_total
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0
        return {
            'portfolio_value': portfolio_value,
            'realized_pnl': float(self.realized_pnl_total),
            'unrealized_pnl': float(unrealized_pnl),
            'total_return': (portfolio_value - self.config.initial_cash) / self.config.initial_cash,
            'num_positions': float(len(self.positions)),
            'cash': float(self.cash),
            'trades': float(self.total_trades),
            'winning_trades': float(self.winning_trades),
            'losing_trades': float(self.losing_trades),
            'win_rate': float(win_rate),
            'stop_loss_exits': float(self.stop_loss_exits),
            'fees_paid': float(self.fees_paid_total),
            'drawdown_pct': (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if self.peak_portfolio_value > 0
            else 0.0,
            'closed_trades': self.closed_trades.copy(),
        }
    
    def _get_observation(self) -> np.ndarray:
        """Build observation state for agent"""
        prices = self.last_prices

        # Normalize prices (log scale for stability)
        price_features = np.log1p(np.maximum(prices, 1e-9)).astype(np.float32)

        # Portfolio state
        portfolio_value = self._get_portfolio_value(prices)
        cash_norm = np.log1p(self.cash / 1e6)
        num_positions = len(self.positions) / self.config.max_positions
        portfolio_return = (portfolio_value - self.config.initial_cash) / self.config.initial_cash

        # Combine into state vector
        obs = np.concatenate([
            price_features,
            np.array([cash_norm, num_positions, portfolio_return, 0.0], dtype=np.float32),
        ]).astype(np.float32)

        return obs
    
    def render(self, mode: str = 'human') -> None:
        """Print current state"""
        portfolio_value = self._get_portfolio_value(self.last_prices)
        print(
            f"Step {self.current_step:6d} | Portfolio: ${portfolio_value:10.2f} | "
            f"Positions: {len(self.positions)} | Cash: ${self.cash:8.2f}"
        )


def _default_symbols() -> List[str]:
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'LINKUSDT', 'LTCUSDT',
    ]


def load_training_data(
    filepath: str,
    num_days: int = 7,
    symbols: Optional[Sequence[str]] = None,
    max_symbols: int = 3,  # Reduced default from 10 for memory efficiency
    use_cache: bool = True,
    cache_dir: str = '.cache',
) -> pd.DataFrame:
    """Memory-efficient parquet loader using symbol filters + local cache.

    The loader reads only required columns and loads each symbol independently,
    which keeps peak RAM low on laptop hardware. Uses only CLOSE prices to minimize
    memory footprint.
    """
    symbol_list = list(symbols) if symbols else _default_symbols()
    symbol_list = symbol_list[:max_symbols]
    minutes = int(num_days) * 24 * 60

    cache_path = (
        Path(cache_dir)
        / f"v3_train_days{num_days}_sym{len(symbol_list)}_{'-'.join(symbol_list)}.parquet"
    )
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    dataset = ds.dataset(filepath, format='parquet')
    frames: List[pd.DataFrame] = []
    # ONLY load close price - reduces memory footprint significantly
    columns = ['symbol', 'open_time', 'close']

    for symbol in symbol_list:
        symbol_filter = ds.field('symbol') == symbol

        # Read one column first to find a recent window without loading full history.
        open_time_table = dataset.to_table(columns=['open_time'], filter=symbol_filter)
        if open_time_table.num_rows == 0:
            continue

        open_times = open_time_table.column('open_time').to_numpy()
        max_open_time = int(open_times.max())

        # Detect parquet timestamp unit by observed minute-step size.
        unit_window = open_times[-min(5000, len(open_times)):]
        diffs = np.diff(unit_window.astype(np.int64))
        diffs = diffs[diffs > 0]
        median_step = int(np.median(diffs)) if diffs.size else 60_000
        if median_step >= 10_000_000_000:
            minute_unit = 60_000_000_000  # ns
        elif median_step >= 10_000_000:
            minute_unit = 60_000_000  # us
        else:
            minute_unit = 60_000  # ms

        cutoff = max_open_time - (minutes * minute_unit)

        table = dataset.to_table(
            columns=columns,
            filter=symbol_filter & (ds.field('open_time') >= cutoff),
        )
        if table.num_rows == 0:
            continue

        part = table.to_pandas()
        part['symbol'] = part['symbol'].astype('category')
        part['close'] = part['close'].astype(np.float32)  # Use float32 for prices
        part['open_time'] = part['open_time'].astype(np.int64)
        frames.append(part)

    if not frames:
        raise ValueError('No rows loaded. Check symbols or parquet path.')

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['symbol', 'open_time']).reset_index(drop=True)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


if __name__ == '__main__':
    # Example: Create env and take random steps
    print("Loading data...")
    data = load_training_data(
        'binance_spot_1m_last4y_single.parquet',
        num_days=3,
        max_symbols=8,
        use_cache=True,
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Symbols: {data['symbol'].nunique()}")
    print("\nInitializing environment...")
    
    env = CryptoTradingEnv(data, config=TradingConfig(
        initial_cash=10_000.0,
        max_positions=5,
        max_budget_per_trade=2_000.0,
        keep_history=False,
    ))
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation (first 5): {obs[:5]}")
    
    print("\nTaking 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if i % 2 == 0:
            env.render()
        if done:
            break
