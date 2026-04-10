"""Minimal RL trading environment with memory-efficient parquet loading."""

from __future__ import annotations

from collections import deque
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
    
    # Reward modeling - ULTRA AGGRESSIVE to force trading
    trade_action_bonus: float = 15.0  # VERY STRONG bonus for taking a trade (buy or sell)
    profit_bonus_scale: float = 200.0  # Bonus multiplier for realized profits
    realized_pnl_bonus: float = 100.0  # STRONG bonus for closing profitable trades
    diversification_bonus: float = 20.0  # Reduced - not main focus
    drawdown_penalty: float = 100.0  # Penalty for losses
    inactivity_penalty: float = -5.0  # EXTREME penalty for holding (do nothing) - every step costs 5
    transaction_cost: float = 0.001  # 0.1% transaction fee
    fee_regime: str = 'legacy'
    buy_fee_rate: Optional[float] = None
    sell_fee_rate: Optional[float] = None
    
    # Risk management
    stop_loss_pct: float = 0.05  # Emergency exit at 5% loss
    max_drawdown_pct: float = 0.20  # Max allowed drawdown from peak
    keep_history: bool = False  # NEVER keep history during training
    max_history_rows: int = 100  # Minimal history if enabled
    debug_actions: bool = False
    invalid_sell_mode: str = 'force_buy'  # force_buy | hold | penalize
    invalid_sell_penalty: float = 3.0
    trade_execution_penalty: float = 0.0
    action_mapping_mode: str = 'legacy'  # legacy | validity_constrained
    action_deadzone: float = 0.15
    trade_rate_window: int = 240
    target_trade_rate: float = 0.18
    trade_rate_penalty: float = 0.0
    min_hold_steps: int = 0
    trade_cooldown_steps: int = 0
    max_trades_per_window: int = 0
    trade_window_steps: int = 240
    constraint_violation_penalty: float = 0.0
    reward_equity_delta_scale: float = 0.0
    turnover_penalty_rate: float = 0.0
    continuous_drawdown_penalty: float = 0.0
    randomize_episode_start: bool = False
    min_episode_steps: int = 0
    max_episode_steps: int = 0
    fee_randomization_pct: float = 0.0


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
        self.hold_actions = 0
        self.buy_actions = 0
        self.sell_actions = 0
        self.invalid_sell_attempts = 0
        self.remapped_actions = 0
        self.recent_trade_actions = deque(maxlen=max(10, int(self.config.trade_rate_window)))
        self.executed_trade_steps = deque()
        self.last_trade_step_by_symbol: Dict[str, int] = {}
        self.last_trade_step_global = -10**9
        self.episode_start_step = 0
        self.episode_end_step = self.n_steps - 1
        self.current_buy_fee_multiplier = 1.0
        self.current_sell_fee_multiplier = 1.0
        self._last_executed_notional = 0.0
        
        # Action space: [action_type, symbol_idx, amount_pct]
        # action_type: 0=hold all, 1=buy, 2=sell
        # symbol_idx: which symbol (0-70)
        # amount_pct: % of available cash (0-1)
        if str(self.config.action_mapping_mode).lower() == 'validity_constrained':
            # intent in [-1, 1], symbol selector in [0, n_symbols-1], amount in [0, 1]
            low = np.array([-1, 0, 0], dtype=np.float32)
            high = np.array([1, len(self.symbols) - 1, 1], dtype=np.float32)
        else:
            low = np.array([0, 0, 0], dtype=np.float32)
            high = np.array([2, len(self.symbols) - 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

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
        self.hold_actions = 0
        self.buy_actions = 0
        self.sell_actions = 0
        self.invalid_sell_attempts = 0
        self.remapped_actions = 0
        self.recent_trade_actions.clear()
        self.executed_trade_steps.clear()
        self.last_trade_step_by_symbol = {}
        self.last_trade_step_global = -10**9

        self._last_executed_notional = 0.0
        fee_jitter = max(0.0, float(self.config.fee_randomization_pct))
        if fee_jitter > 0:
            low = max(0.01, 1.0 - fee_jitter)
            high = 1.0 + fee_jitter
            self.current_buy_fee_multiplier = float(self.np_random.uniform(low, high))
            self.current_sell_fee_multiplier = float(self.np_random.uniform(low, high))
        else:
            self.current_buy_fee_multiplier = 1.0
            self.current_sell_fee_multiplier = 1.0

        self.episode_start_step = 0
        self.episode_end_step = self.n_steps - 1
        if bool(self.config.randomize_episode_start):
            max_len_cfg = int(self.config.max_episode_steps)
            min_len_cfg = int(self.config.min_episode_steps)
            max_len = self.n_steps if max_len_cfg <= 0 else min(self.n_steps, max_len_cfg)
            min_len = max(2, min_len_cfg if min_len_cfg > 0 else max_len)
            min_len = min(min_len, max_len)

            episode_len = int(self.np_random.integers(min_len, max_len + 1))
            max_start = max(0, self.n_steps - episode_len)
            self.episode_start_step = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
            self.episode_end_step = self.episode_start_step + episode_len - 1
            self.current_step = self.episode_start_step
            self.last_prices = self.price_matrix[self.current_step]

        return self._get_observation(), {}

    def _prune_trade_window(self) -> None:
        window = max(1, int(self.config.trade_window_steps))
        cutoff = self.current_step - window
        while self.executed_trade_steps and self.executed_trade_steps[0] <= cutoff:
            self.executed_trade_steps.popleft()

    def _can_execute_trade(self, symbol: str, action_type: int) -> Tuple[bool, str]:
        self._prune_trade_window()

        max_window = int(self.config.max_trades_per_window)
        if max_window > 0 and len(self.executed_trade_steps) >= max_window:
            return False, 'trade_window_cap'

        cooldown = int(self.config.trade_cooldown_steps)
        if cooldown > 0 and (self.current_step - self.last_trade_step_global) < cooldown:
            return False, 'cooldown'

        # Enforce minimum holding horizon before explicit sell actions.
        if action_type == 2 and symbol in self.positions:
            min_hold = int(self.config.min_hold_steps)
            held = self.current_step - int(self.positions[symbol]['entry_step'])
            if min_hold > 0 and held < min_hold:
                return False, 'min_hold'

        return True, ''

    def _effective_buy_fee_rate(self) -> float:
        if self.config.buy_fee_rate is not None:
            return float(self.config.buy_fee_rate) * float(self.current_buy_fee_multiplier)
        return float(self.config.transaction_cost) * float(self.current_buy_fee_multiplier)

    def _effective_sell_fee_rate(self) -> float:
        if self.config.sell_fee_rate is not None:
            return float(self.config.sell_fee_rate) * float(self.current_sell_fee_multiplier)
        return float(self.config.transaction_cost) * float(self.current_sell_fee_multiplier)

    def _resolve_action_components(self, action: np.ndarray) -> Tuple[int, int, float]:
        mode = str(self.config.action_mapping_mode).lower()
        if mode != 'validity_constrained':
            action_type = int(np.clip(action[0], 0, 2))
            symbol_idx = int(np.clip(action[1], 0, len(self.symbols) - 1))
            amount_pct = float(np.clip(action[2], 0, 1))
            return action_type, symbol_idx, amount_pct

        intent = float(np.clip(action[0], -1, 1))
        symbol_idx = int(np.clip(action[1], 0, len(self.symbols) - 1))
        amount_pct = float(np.clip(action[2], 0, 1))
        deadzone = max(0.0, float(self.config.action_deadzone))

        if abs(intent) <= deadzone:
            action_type = 0
        elif intent > deadzone:
            action_type = 1
        else:
            action_type = 2 if len(self.positions) > 0 else 0

        return action_type, symbol_idx, amount_pct
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading action and return new state + reward
        
        Args:
            action: [action_type (0-2), symbol_idx (0-70), amount_pct (0-1)]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        prev_portfolio_value = self._get_portfolio_value(self.last_prices)
        self.current_step += 1
        step_reward = 0.0
        action_taken = False  # Track if a real action (buy/sell) was taken
        self._last_executed_notional = 0.0
        
        # Limit to available steps in data
        if self.current_step >= self.n_steps or self.current_step > self.episode_end_step:
            return self._get_observation(), step_reward, True, False, {}

        if self.config.debug_actions and self.current_step < 3:
            print(f"[DEBUG] step={self.current_step} RAW ACTION: {action}")
        
        prices = self.price_matrix[self.current_step]
        self.last_prices = prices
        
        # Parse action
        action_type, symbol_idx, amount_pct = self._resolve_action_components(action)
        
        # IMPORTANT: If model is trying to trade (action_type > 0), force a minimum amount
        # This prevents model from "intending" to trade but using 0 amount
        if action_type > 0 and amount_pct < 0.05:
            amount_pct = 0.1  # Force meaningful trade size

        # Validity gate: selling with no open positions is invalid.
        # Remap invalid sell actions to buy so impossible actions still explore the market.
        if action_type == 2 and len(self.positions) == 0:
            self.invalid_sell_attempts += 1
            self.remapped_actions += 1
            mode = str(self.config.invalid_sell_mode).lower()
            if mode == 'force_buy':
                action_type = 1
            elif mode == 'hold':
                action_type = 0
            elif mode == 'penalize':
                pass
            else:
                action_type = 1
            step_reward -= abs(float(self.config.invalid_sell_penalty))
        
        symbol = self.symbols[symbol_idx]
        current_price = float(prices[symbol_idx])

        # Hard execution constraints to reduce pathological churn.
        if action_type in (1, 2) and amount_pct > 0.05:
            can_execute, _ = self._can_execute_trade(symbol, action_type)
            if not can_execute:
                self.remapped_actions += 1
                action_type = 0
                if self.config.constraint_violation_penalty > 0:
                    step_reward -= float(self.config.constraint_violation_penalty)
        
        # Track final (possibly remapped) action type for diagnostics.
        if action_type == 0:
            self.hold_actions += 1
        elif action_type == 1:
            self.buy_actions += 1
        elif action_type == 2:
            self.sell_actions += 1

        # Execute trading action and give bonus for attempting to trade
        if action_type == 1 and amount_pct > 0.05:  # BUY (lowered threshold from 0.1)
            step_reward += self._execute_buy(symbol, current_price, amount_pct)
            action_taken = True
        elif action_type == 2 and amount_pct > 0.05:  # SELL (lowered threshold from 0.1)
            sell_reward = self._execute_sell(symbol, current_price, amount_pct)
            if sell_reward > 0:  # Actual position was sold
                step_reward += sell_reward
                action_taken = True
            else:  # Invalid sell (no position to sell)
                step_reward -= 3.0  # HEAVY penalty for invalid action
        else:
            # PENALTY for inactivity (doing nothing / hold action)
            step_reward += self.config.inactivity_penalty

        self.recent_trade_actions.append(1 if action_taken else 0)

        if action_taken:
            self.executed_trade_steps.append(self.current_step)
            self.last_trade_step_global = self.current_step
            self.last_trade_step_by_symbol[symbol] = self.current_step

        if action_taken and self.config.trade_execution_penalty > 0:
            step_reward -= float(self.config.trade_execution_penalty)

        if action_taken and self.config.turnover_penalty_rate > 0 and self._last_executed_notional > 0:
            notional_ratio = self._last_executed_notional / max(1e-6, float(self.config.initial_cash))
            step_reward -= float(self.config.turnover_penalty_rate) * float(notional_ratio)

        # Explicit churn regularization to discourage near-every-step trading.
        if self.config.trade_rate_penalty > 0 and len(self.recent_trade_actions) >= 10:
            recent_rate = float(sum(self.recent_trade_actions)) / float(len(self.recent_trade_actions))
            target_rate = max(0.0, min(1.0, float(self.config.target_trade_rate)))
            if recent_rate > target_rate:
                step_reward -= float(self.config.trade_rate_penalty) * (recent_rate - target_rate)
        
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

        if self.config.reward_equity_delta_scale != 0:
            delta_ratio = (portfolio_value - prev_portfolio_value) / max(1e-6, float(self.config.initial_cash))
            step_reward += float(self.config.reward_equity_delta_scale) * float(delta_ratio)
        
        # Remove the passive profit bonus - force trading for profits
        # Profit bonus (only if profitable and we have active positions)
        if len(self.positions) > 0:
            unrealized_pnl = portfolio_value - self.config.initial_cash
            if unrealized_pnl > 0:
                step_reward += unrealized_pnl / 200.0  # Much smaller bonus
        
        # Drawdown penalty
        drawdown_pct = 0.0
        if portfolio_value < self.peak_portfolio_value:
            drawdown_pct = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown_pct > self.config.max_drawdown_pct:
                step_reward -= self.config.drawdown_penalty
        if self.config.continuous_drawdown_penalty > 0 and drawdown_pct > 0:
            step_reward -= float(self.config.continuous_drawdown_penalty) * float(drawdown_pct)
        
        # Diversification bonus - encourage multiple positions
        if len(self.positions) > 1:
            step_reward += len(self.positions) * 0.1  # Reduced bonus
        
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
        
        done = self.current_step >= self.episode_end_step
        
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
            'hold_actions': self.hold_actions,
            'buy_actions': self.buy_actions,
            'sell_actions': self.sell_actions,
            'invalid_sell_attempts': self.invalid_sell_attempts,
            'remapped_actions': self.remapped_actions,
            'recent_trade_rate': float(sum(self.recent_trade_actions)) / float(len(self.recent_trade_actions))
            if len(self.recent_trade_actions) > 0
            else 0.0,
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
        fee_rate = self._effective_buy_fee_rate()
        amount_after_fee = amount * (1 - fee_rate)
        fee = amount * fee_rate
        qty = amount_after_fee / price
        
        self.positions[symbol] = {
            'qty': qty,
            'entry_price': price,
            'entry_step': self.current_step
        }
        self.cash -= amount
        self._last_executed_notional = float(amount)
        self.fees_paid_total += fee
        self.total_trades += 1
        
        # STRONG bonus for taking a trade action (buy)
        trade_bonus = self.config.trade_action_bonus  # 5.0 - strong signal
        fee_penalty = fee_rate * amount
        return trade_bonus - fee_penalty
    
    def _execute_sell(self, symbol: str, price: float, pct: float) -> float:
        """Execute sell order"""
        if symbol not in self.positions:
            return 0
        
        # Sell portion or all
        pos = self.positions[symbol]
        qty_to_sell = pos['qty'] * pct if pct < 1.0 else pos['qty']
        
        gross_proceeds = qty_to_sell * price
        fee_rate = self._effective_sell_fee_rate()
        fee = gross_proceeds * fee_rate
        proceeds = gross_proceeds - fee
        self.cash += proceeds
        self._last_executed_notional = float(gross_proceeds)
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
        
        # Reward = strong bonus for selling + realized P&L
        sell_action_bonus = self.config.trade_action_bonus  # 5.0
        if realized_pnl > 0:
            profit_bonus = min(realized_pnl * 10, self.config.realized_pnl_bonus)  # Scale up small profits
            return sell_action_bonus + profit_bonus
        else:
            # Even losses get the action bonus to encourage trying
            loss_penalty = realized_pnl / 10.0  # Small loss scaling
            return sell_action_bonus + loss_penalty

    def _close_position(self, symbol: str, price: float, reason: str = 'forced') -> None:
        """Force-close a position (stop loss or max duration)"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            gross_proceeds = pos['qty'] * price
            fee_rate = self._effective_sell_fee_rate()
            fee = gross_proceeds * fee_rate
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
        total_actions = self.hold_actions + self.buy_actions + self.sell_actions
        actions_den = float(total_actions) if total_actions > 0 else 1.0
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
            'hold_actions': float(self.hold_actions),
            'buy_actions': float(self.buy_actions),
            'sell_actions': float(self.sell_actions),
            'invalid_sell_attempts': float(self.invalid_sell_attempts),
            'remapped_actions': float(self.remapped_actions),
            'hold_action_pct': float(self.hold_actions) / actions_den,
            'buy_action_pct': float(self.buy_actions) / actions_den,
            'sell_action_pct': float(self.sell_actions) / actions_den,
            'invalid_sell_pct': float(self.invalid_sell_attempts) / actions_den,
            'remapped_action_pct': float(self.remapped_actions) / actions_den,
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
