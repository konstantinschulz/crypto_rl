import logging
from typing import Any

import numpy as np

from crypto_rl.env.action_processing import (
    apply_continuous_action,
    apply_discrete_action,
)
from crypto_rl.env.logging_utils import log_action
from crypto_rl.env.metrics import get_per_asset_summary


def step_env(env, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Execute one environment step for MinimalCryptoEnv."""
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
    if env.action_space_type == "continuous":
        # Snapshot old asset exposure fraction before rebalancing (for logging)
        _pre_asset_value = np.sum(env.holdings * current_prices)
        _pre_port = env.cash + _pre_asset_value
        _old_asset_frac = (
            (_pre_asset_value / _pre_port) if _pre_port > 1e-8 else 0.0
        )
        # Continuous action processing moved to helper
        fee_paid, trade_units = apply_continuous_action(env, action)
    elif env.action_space_type == "multidiscrete":
        # Safely copy the action so we don't mutate Gym's read-only array
        mod_action = np.copy(action)
        # action[2] is 0-100. Convert to a 0.0 - 1.0 fraction
        amount_pct = mod_action[2] / 100.0
        # Force a HOLD if the requested amount is lower than the dead zone
        if amount_pct < env.action_dead_zone:
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
        "hold_incentive": 0.0,
        "terminal_return": 0.0,
        "turnover_penalty": 0.0,  # NEW: Excessive turnover penalty tracker
    }
    # --- CALCULATE PER-ASSET MULTI-TRADE METRICS ---
    deltas = env.holdings - old_holdings
    for i, delta in enumerate(deltas):
        # 1. BOUGHT (Scaled in)
        if delta > 1e-8:
            # --- NEW: Record entry step if establishing a new position ---
            if old_holdings[i] < 1e-8:
                env.entry_step[i] = env.current_step
            # Cost of newly acquired units (Delta is the number of units)
            cost_of_new_with_fees = (
                delta * current_prices[i] * (1.0 + env.fee_rate)
            )
            # Value of existing units AT INITIAL COST BASIS
            value_of_existing = old_holdings[i] * env.avg_entry_price[i]
            # Update weighted average entry price
            env.avg_entry_price[i] = (
                value_of_existing + cost_of_new_with_fees
            ) / env.holdings[i]

        # 2. SOLD (Scaled out)
        elif delta < -1e-8:
            env.total_closed_trades += 1
            env.per_asset_trades[i] += 1
            # SAFEGUARD: Clamp amount sold to what we actually owned to kill phantom PnL
            amount_sold = min(abs(delta), old_holdings[i])
            # Real revenue generated minus fees
            revenue = amount_sold * current_prices[i] * (1.0 - env.fee_rate)
            # Cost basis (Do NOT multiply by 1 + fee_rate again; avg_entry_price already includes it)
            cost_basis = amount_sold * env.avg_entry_price[i]
            adjusted_pnl = revenue - cost_basis
            env.per_asset_realized_pnl[i] += adjusted_pnl
            if cost_basis > 1e-8:
                trade_return = (revenue - cost_basis) / cost_basis
                # --- NEW: Excessive Turnover Penalty Logic ---
                holding_period = env.current_step - env.entry_step[i]
                # Penalize positions closed in under 15 minutes (or steps)
                if holding_period < env.config.turnover_penalty_steps_threshold:
                    reward_components["turnover_penalty"] -= (
                        env.config.turnover_penalty
                    )
                # A win requires net revenue to exceed the exact cost we paid for those units
                if revenue > cost_basis:
                    env.winning_trades_count += 1
                    env.per_asset_wins[i] += 1
                # --- NEW: Dynamic Profit Bonus Scaling ---
                # 1. Time multiplier: Cap at 2.0x for holding 4 hours (240 steps)
                time_mult = 1.0 + min(1.0, holding_period / 240.0)
                # 2. HTF Slope multiplier: Scale up by 1h momentum magnitude
                slope_mag = 0.0
                if env.htf_slope_1h_df is not None and env.current_step < len(
                    env.htf_slope_1h_df
                ):
                    slope_mag = abs(env.htf_slope_1h_df.iat[env.current_step, i])
                # Multiply by an arbitrary scalar (10.0) to make the slope meaningful in the equation
                slope_mult = 1.0 + (slope_mag * 10.0)
                scaled_profit_bonus = env.profit_bonus * time_mult * slope_mult
                # Symmetric scaling: rewards profits, penalizes losses proportionally
                reward_components["profit_bonus"] += (
                    scaled_profit_bonus * trade_return
                )
            # If we fully closed out, reset cost basis to 0
            if env.holdings[i] < 1e-8:
                env.avg_entry_price[i] = 0.0
                # SAFEGUARD: Snap negative holdings to 0 to kill the short-selling exploit
                env.holdings[i] = 0.0
                env.entry_step[i] = 0  # Reset entry step tracker
    # Advance step
    env.current_step += 1
    done = env.current_step >= env.prices_arr.shape[0]
    current_asset_value = np.sum(env.holdings * next_prices)
    env.portfolio_value = env.cash + current_asset_value
    env.peak_portfolio_value = max(env.peak_portfolio_value, env.portfolio_value)
    # Range: 0.0 (at peak) down to -1.0 (-100% loss)
    current_drawdown = (
        env.portfolio_value - env.peak_portfolio_value
    ) / env.peak_portfolio_value
    # Calculate delta. If current (-0.10) is worse than previous (-0.05), delta is -0.05.
    # We use min(0, ...) to ensure we ONLY capture worsening drawdowns, ignoring recoveries.
    delta_drawdown = min(0.0, current_drawdown - env.previous_drawdown)
    # Update previous drawdown for the next step
    env.previous_drawdown = current_drawdown
    # ==========================================
    # NEW: 1. Hold Cost / Inactivity Penalty
    # ==========================================
    underwater_penalty = 0.0
    for i in range(env.num_assets):
        if env.holdings[i] > 1e-8 and env.avg_entry_price[i] > 1e-8:
            unrealized_pnl_pct = (
                next_prices[i] - env.avg_entry_price[i]
            ) / env.avg_entry_price[i]
            # If position is down more than 5%, apply the hold_cost_rate as a recurring penalty
            if unrealized_pnl_pct < -0.05:
                underwater_penalty += env.hold_cost_rate
    reward_components["hold_cost"] = -underwater_penalty
    # ==========================================
    # NEW: 2. Explicit Fee Penalty in Reward
    # ==========================================
    # Normalize fee relative to portfolio size so it scales correctly with returns
    fee_penalty_pct = (
        (fee_paid / prev_portfolio_value) if prev_portfolio_value > 1e-8 else 0.0
    )
    reward_components["fee_penalty"] = -fee_penalty_pct  # Alpha = 1.0 multiplier
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

    if env.reward_type == "excess_return":
        alpha_diff = portfolio_return - market_return
        # Slightly penalize underperformance, but avoid the 200x asymmetric distortion
        if alpha_diff < 0:
            alpha_diff *= 1.2
        reward_components["market_alpha"] = alpha_diff
        reward_components["drawdown_penalty"] = (
            delta_drawdown * env.drawdown_penalty_coef
        )
    else:
        reward_components["market_alpha"] = portfolio_return
    # --- ACTION-SPECIFIC INCENTIVES (Optional) ---
    if env.action_space_type == "continuous":
        n_held = sum(
            1
            for i in range(env.num_assets)
            if abs(action[i]) <= env.action_dead_zone
        )
        reward_components["hold_incentive"] = n_held * env.hold_incentive
    info: dict[str, Any] = {}
    if done:  # this episode has finished, so we need to liquidate all remaining holdings
        liquidation_revenue = 0.0
        liquidation_fees = 0.0
        for i in range(env.num_assets):
            if env.holdings[i] > 1e-8:
                amount_sold = env.holdings[i]
                # Calculate liquidation revenue and fees
                revenue = amount_sold * next_prices[i] * (1.0 - env.fee_rate)
                fee = amount_sold * next_prices[i] * env.fee_rate
                # Apply the corrected cost basis with entry drag
                cost_basis = (
                    amount_sold * env.avg_entry_price[i] * (1.0 + env.fee_rate)
                )
                adjusted_pnl = revenue - cost_basis
                env.per_asset_realized_pnl[i] += adjusted_pnl
                env.per_asset_fees[i] += fee
                liquidation_revenue += revenue
                liquidation_fees += fee
                env.total_closed_trades += 1
                env.per_asset_trades[i] += 1
                if cost_basis > 1e-8 and adjusted_pnl > 0:
                    env.winning_trades_count += 1
                    env.per_asset_wins[i] += 1

                # Wipe assets from state
                env.holdings[i] = 0.0
                env.avg_entry_price[i] = 0.0
                env.entry_step[i] = 0
        # Convert portfolio entirely to cash based on liquidation
        env.cash += liquidation_revenue
        env.fees_paid_total += liquidation_fees
        env.portfolio_value = env.cash  # Holdings are 0, PV is just cash

        terminal_return = (
            env.portfolio_value - env.config.budget_initial
        ) / env.config.budget_initial
        reward_components["terminal_return"] = terminal_return
        # Capture the final state right before the auto-reset
        info["per_asset_stats"] = get_per_asset_summary(env)
        info["final_portfolio_value"] = env.portfolio_value
        info["final_trades_count"] = env.trades_count
        info["final_fees_paid"] = env.fees_paid_total

        sum_per_asset_pnl = np.sum(env.per_asset_realized_pnl)
        actual_pnl = env.portfolio_value - env.config.budget_initial

        # Log warning if discrepancy exceeds $0.05
        if abs(sum_per_asset_pnl - actual_pnl) > 0.05:
            logging.warning(
                f"PnL Mismatch Detected! Portfolio PnL: ${actual_pnl:.2f} vs "
                f"Sum Per-Asset PnL: ${sum_per_asset_pnl:.2f}"
            )
    if step_penalty >= 0.1:
        logging.debug(f"High step penalty value (>= 0.1): {step_penalty}")

    # Explicitly clip raw rewards at the source to prevent variance explosion
    reward = np.clip(sum(reward_components.values()), -1.0, 1.0)
    if not env.disable_logging:
        if env.action_space_type == "continuous":
            exp_act = np.exp(action - np.max(action))
            weights = exp_act / np.sum(exp_act)
            # Determine net direction of this rebalance for logging purposes.
            # Compare new asset-exposure fraction to the pre-trade snapshot taken
            # at the top of step().  Positive delta → buying assets (BUY=1),
            # negative delta → reducing assets (SELL=2), flat → HOLD=0.
            _new_asset_value = np.sum(env.holdings * next_prices)
            _new_port = env.cash + _new_asset_value
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
                env,
                env.current_step,
                eff_action,
                reward,
                prev_portfolio_value,
                fee=fee_paid,
                reward_components=reward_components,
            )
        elif env.action_space_type == "multidiscrete":
            # Discrete action logging remains unchanged (action variables are updated inside helper)
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
