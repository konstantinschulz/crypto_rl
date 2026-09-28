from __future__ import annotations  # avoid circular import issues with TYPE_CHECKING

from typing import TYPE_CHECKING, Any

import numpy as np

from crypto_rl.env.action_processing import (
    apply_discrete_action,
)
from crypto_rl.env.logging_utils import log_action
from crypto_rl.env.metrics import get_per_asset_summary

if TYPE_CHECKING:  # avoid circular import issues
    from crypto_rl.env.minimal_env import MinimalCryptoEnv


def _process_buy_accounting(
    env,
    asset_idx: int,
    amount_bought: float,
    trade_price: float,
    old_holding: float,
):
    """
    Core accounting logic for opening or scaling into positions.
    Updates VWAP entry prices and volume-weighted entry steps.
    """
    if old_holding < 1e-8:
        env.entry_step[asset_idx] = env.current_step
    else:
        old_weight = old_holding / env.holdings[asset_idx]
        new_weight = amount_bought / env.holdings[asset_idx]
        env.entry_step[asset_idx] = int(
            (env.entry_step[asset_idx] * old_weight) + (env.current_step * new_weight)
        )
    cost_of_new_with_fees = amount_bought * trade_price * (1.0 + env.fee_rate)
    value_of_existing = old_holding * env.avg_entry_price[asset_idx]

    env.avg_entry_price[asset_idx] = (
        value_of_existing + cost_of_new_with_fees
    ) / env.holdings[asset_idx]


def _process_sell_accounting(
    env,
    asset_idx: int,
    amount_sold: float,
    trade_price: float,
    prev_portfolio_value: float,
    reward_components: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    Core accounting logic for closing positions.
    Calculates PnL, updates win rates, and applies shaped rewards.
    Used seamlessly during both in-episode sells and terminal liquidations.
    """
    env.total_closed_trades += 1
    env.per_asset_trades[asset_idx] += 1

    revenue = amount_sold * trade_price * (1.0 - env.fee_rate)
    fee = amount_sold * trade_price * env.fee_rate

    # Safely compute cost basis (entry drag is already baked into avg_entry_price)
    cost_basis = amount_sold * env.avg_entry_price[asset_idx]

    adjusted_pnl = revenue - cost_basis
    env.per_asset_realized_pnl[asset_idx] += adjusted_pnl

    if cost_basis > 1e-8:
        trade_return = (revenue - cost_basis) / cost_basis
        trade_weight = (
            cost_basis / prev_portfolio_value if prev_portfolio_value > 1e-8 else 0.0
        )

        # A win requires net revenue to exceed the exact cost we paid for those units
        if revenue > cost_basis:
            env.winning_trades_count += 1
            env.per_asset_wins[asset_idx] += 1

        # Only calculate RL reward bonuses if we are actively stepping (not liquidating)
        if reward_components is not None:
            holding_period = env.current_step - env.entry_step[asset_idx]
            # Reward cutting losses early instead of holding to the bottom
            if trade_return < 0 and holding_period > 15:
                # Small positive reinforcement for taking the loss and freeing up cash
                reward_components["loss_cut_bonus"] = (
                    abs(trade_return) * env.config.loss_cut_bonus
                )
            # Turnover Penalty
            if holding_period < env.config.turnover_penalty_steps_threshold:
                reward_components["turnover_penalty"] -= (
                    env.config.turnover_penalty * trade_weight
                )

            # Dynamic Profit Bonus Scaling
            time_mult = 1.0 + min(1.0, holding_period / 240.0)
            slope_mag = 0.0
            if env.htf_slope_1h_df is not None and env.current_step < len(
                env.htf_slope_1h_df
            ):
                slope_mag = abs(env.htf_slope_1h_df.iat[env.current_step, asset_idx])

            slope_mult = 1.0 + (slope_mag * 10.0)
            scaled_profit_bonus = env.config.profit_bonus * time_mult * slope_mult

            reward_components["profit_bonus"] += (
                scaled_profit_bonus * trade_return * trade_weight
            )

    return revenue, fee


def step_env(
    env: MinimalCryptoEnv, action
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Execute one environment step for MinimalCryptoEnv.
    Returns:
        obs: np.ndarray - The next observation after the step.
        reward: float - The reward obtained from the step.
        done: bool - Whether the episode has ended.
        truncated: bool - Whether the episode was truncated (not used here).
        info: dict - Additional information about the step, including reward components.
    """
    prev_portfolio_value = env.portfolio_value
    current_prices = env.prices_arr[env.current_step - 1]
    next_prices = env.prices_arr[env.current_step]

    fee_paid = 0.0
    # Retrieve any step penalty set by discrete action processing
    step_penalty = getattr(env, "_step_penalty", 0.0)
    realised_pnl = 0.0

    is_valid_sell = False
    env.last_remap_note = None
    # --- SNAPSHOT HOLDINGS BEFORE ACTION ---
    old_holdings = np.copy(env.holdings)
    if env.config.action_space_type == "multidiscrete":
        # Safely copy the action so we don't mutate Gym's read-only array
        mod_action = np.copy(action)
        # action[2] is 0-100. Convert to a 0.0 - 1.0 fraction
        amount_pct = mod_action[2] / 100.0
        # Force a HOLD if the requested amount is lower than the dead zone
        if amount_pct < env.config.action_dead_zone:
            mod_action[0] = 0  # 0 = Hold
        # Ensure we use mod_action for logging later
        action = mod_action
        # Discrete action processing moved to helper
        fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price = (
            apply_discrete_action(env, action)
        )
        asset_idx = action[1]
        if fee_paid > 0:
            env.per_asset_fees[asset_idx] += fee_paid

    # 1. Initialize a decomposition tracker
    reward_components = {
        "market_alpha": 0.0,
        "hold_cost": 0.0,
        "profit_bonus": 0.0,
        "drawdown_penalty": 0.0,
        "rule_penalties": -step_penalty,  # From invalid buys/sells
        "terminal_return": 0.0,
        "turnover_penalty": 0.0,
        "capital_preservation_bonus": 0.0,
    }

    # --- CALCULATE PER-ASSET MULTI-TRADE METRICS ---
    deltas = env.holdings - old_holdings
    for i, delta in enumerate(deltas):
        # 1. BOUGHT (Scaled in)
        if delta > 1e-8:
            _process_buy_accounting(env, i, delta, current_prices[i], old_holdings[i])

        # 2. SOLD (Scaled out)
        elif delta < -1e-8:
            amount_sold = min(abs(delta), old_holdings[i])
            _process_sell_accounting(
                env,
                i,
                amount_sold,
                current_prices[i],
                prev_portfolio_value,
                reward_components,
            )

            # If we fully closed out, cleanup state
            if env.holdings[i] < 1e-8:
                env.avg_entry_price[i] = 0.0
                env.holdings[i] = 0.0
                env.entry_step[i] = 0

    # Advance step
    env.current_step += 1
    # TRIGGER EPISODE END BASED ON THE RANDOMIZED BOUNDARY
    done = env.current_step >= env.max_steps
    current_asset_value = np.sum(env.holdings * next_prices)
    env.portfolio_value = env.cash + current_asset_value
    env.peak_portfolio_value = max(env.peak_portfolio_value, env.portfolio_value)

    # Range: 0.0 (at peak) down to -1.0 (-100% loss)
    current_drawdown = (
        env.portfolio_value - env.peak_portfolio_value
    ) / env.peak_portfolio_value
    delta_drawdown = min(0.0, current_drawdown - env.previous_drawdown)
    env.previous_drawdown = current_drawdown
    safe_port_val = max(env.portfolio_value, 1e-8)
    # ==========================================
    # 1. Hold Cost / Inactivity Penalty
    # ==========================================
    for i in range(env.num_assets):
        if env.holdings[i] > 1e-8 and env.avg_entry_price[i] > 1e-8:
            unrealized_pnl_pct = (
                next_prices[i] - env.avg_entry_price[i]
            ) / env.avg_entry_price[i]
            if unrealized_pnl_pct < -abs(env.config.hold_penalty_threshold):
                # Position weight relative to total portfolio
                position_weight = (env.holdings[i] * next_prices[i]) / safe_port_val
                # Penalty ramps up as drawdown deepens
                loss_severity = abs(unrealized_pnl_pct)
                hold_penalty = (
                    loss_severity * env.config.hold_cost_rate * 100.0 * position_weight
                )
                reward_components["hold_cost"] -= hold_penalty

    # ==========================================
    # 2. Explicit Fee Penalty in Reward
    # ==========================================
    fee_penalty_pct = (
        (fee_paid / prev_portfolio_value) if prev_portfolio_value > 1e-8 else 0.0
    )
    reward_components["fee_penalty"] = -fee_penalty_pct

    # Reward calculation
    portfolio_return = (
        (env.portfolio_value - prev_portfolio_value) / prev_portfolio_value
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

    if env.config.reward_type == "excess_return":
        # Benchmark Floor: In bear markets (market_return < 0),
        # the baseline switches to 0.0 (Cash), requiring non-negative return for positive alpha.
        effective_benchmark = max(0.0, market_return)
        alpha_diff = portfolio_return - effective_benchmark
        if alpha_diff < 0:
            alpha_diff *= 1.2
        reward_components["market_alpha"] = alpha_diff
        reward_components["drawdown_penalty"] = (
            delta_drawdown * env.config.drawdown_penalty_coef
        )
    else:
        reward_components["market_alpha"] = portfolio_return

    # ==========================================
    # 3. Macro Cash Preference in Bear Regimes
    # ==========================================
    # Fetch current timestep t bounded to valid precalc matrix bounds
    t = min(env.current_step, len(env.precalc_static_obs) - 1)

    # Safely retrieve the bear index using the mapping from feature_utils
    bear_idx = getattr(env, "macro_idx", {}).get("btc_bear", 4)
    btc_is_bear = env.precalc_static_obs[t, bear_idx] > 0.5

    if btc_is_bear:
        cash_ratio = env.cash / safe_port_val
        # Grant micro-bonus for holding >75% in cash during macro bear regimes
        if cash_ratio > 0.75:
            reward_components["capital_preservation_bonus"] = (
                env.config.capital_preservation_bonus
            )

    info: dict[str, Any] = {}

    # ==========================================
    # TERMINAL LIQUIDATION BLOCK
    # ==========================================
    if done:
        liquidation_revenue = 0.0
        liquidation_fees = 0.0
        for i in range(env.num_assets):
            if env.holdings[i] > 1e-8:
                amount_sold = env.holdings[i]

                # Execute terminal sell through the unified helper (no rewards applied)
                revenue, fee = _process_sell_accounting(
                    env,
                    i,
                    amount_sold,
                    next_prices[i],
                    prev_portfolio_value,
                    reward_components=None,
                )

                env.per_asset_fees[i] += fee
                liquidation_revenue += revenue
                liquidation_fees += fee

                # Wipe assets from state
                env.holdings[i] = 0.0
                env.avg_entry_price[i] = 0.0
                env.entry_step[i] = 0

        # Convert portfolio entirely to cash based on liquidation
        env.cash += liquidation_revenue
        env.fees_paid_total += liquidation_fees
        env.portfolio_value = env.cash

        terminal_return = (
            env.portfolio_value - env.config.budget_initial
        ) / env.config.budget_initial
        reward_components["terminal_return"] = terminal_return

        info["per_asset_stats"] = get_per_asset_summary(env)
        info["final_portfolio_value"] = env.portfolio_value
        info["final_trades_count"] = env.trades_count
        info["final_fees_paid"] = env.fees_paid_total

        sum_per_asset_pnl = np.sum(env.per_asset_realized_pnl)
        actual_pnl = env.portfolio_value - env.config.budget_initial

        # Log warning if discrepancy exceeds $0.05
        if abs(sum_per_asset_pnl - actual_pnl) > 0.05:
            env.logger.debug(
                f"PnL Mismatch Detected! Portfolio PnL: ${actual_pnl:.2f} vs "
                f"Sum Per-Asset PnL: ${sum_per_asset_pnl:.2f}"
            )

    if step_penalty >= 0.1:
        env.logger.debug(f"High step penalty value (>= 0.1): {step_penalty}")

    # Explicitly clip raw rewards at the source to prevent variance explosion
    reward = np.clip(sum(reward_components.values()), -1.0, 1.0)

    if (
        not env.config.disable_logging
        and env.config.action_space_type == "multidiscrete"
    ):
        log_action(
            env,
            env.current_step,
            action,
            reward,
            prev_portfolio_value,
            trade_price=trade_price,
            trade_units=trade_units,
            fee=fee_paid,
            reward_components=reward_components,
        )

    info |= {
        "fees_paid": env.fees_paid_total,
        "trades_count": env.trades_count,
        "total_closed_trades": env.total_closed_trades,
        "winning_trades_count": env.winning_trades_count,
        "realised_pnl": realised_pnl,
        "is_valid_sell": is_valid_sell,
        "episode_count": env.episode_count,
        "reward_components": reward_components,
    }

    return (
        env._get_obs(),
        float(reward),
        done,
        False,
        info,
    )
