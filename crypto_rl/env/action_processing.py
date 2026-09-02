import gymnasium
import numpy as np


def apply_continuous_action(env, action):
    """Process continuous actions.

    Returns a tuple (fee_paid, trade_units). ``trade_units`` is set to 0
    because continuous actions rebalance the whole portfolio rather than
    tracking a single trade quantity.
    """
    exp_action = np.exp(action - np.max(action))
    target_weights = exp_action / np.sum(exp_action)

    current_prices = env.prices_arr[env.current_step - 1]
    safe_prices = np.nan_to_num(current_prices, nan=0.0, posinf=0.0, neginf=0.0)
    current_asset_values = env.holdings * safe_prices
    curr_portfolio_val = env.cash + np.sum(current_asset_values)

    old_weights = np.zeros(env.num_assets + 1, dtype=np.float32)
    if curr_portfolio_val > 1e-8:
        old_weights[0] = env.cash / curr_portfolio_val
        old_weights[1:] = np.divide(
            current_asset_values,
            curr_portfolio_val,
            out=np.zeros_like(current_asset_values),
            where=curr_portfolio_val > 1e-8,
        )
    else:
        old_weights[0] = 1.0

    turnover = np.sum(np.abs(target_weights - old_weights))
    min_turnover_threshold = getattr(env, "min_turnover_threshold", 0.02)
    if turnover < min_turnover_threshold:
        return 0.0, 0.0

    rebalance_cost = env.fee_rate * turnover * curr_portfolio_val / 2.0
    new_portfolio_val = max(0.0, curr_portfolio_val - rebalance_cost)

    env.cash = new_portfolio_val * target_weights[0]
    new_asset_allocations = new_portfolio_val * target_weights[1:]
    new_holdings = np.divide(
        new_asset_allocations,
        current_prices,
        out=np.zeros_like(new_asset_allocations),
        where=current_prices > 1e-8,
    )

    for i in range(env.num_assets):
        if new_holdings[i] > env.holdings[i]:
            added_units = new_holdings[i] - env.holdings[i]
            added_cost = added_units * current_prices[i]
            env.total_cost_basis[i] += added_cost
            env.avg_entry_price[i] = (
                env.total_cost_basis[i] / new_holdings[i]
                if new_holdings[i] > 0
                else 0.0
            )
        elif new_holdings[i] < env.holdings[i]:
            if new_holdings[i] <= 1e-9:
                env.avg_entry_price[i] = 0.0
                env.total_cost_basis[i] = 0.0
            else:
                env.total_cost_basis[i] *= new_holdings[i] / env.holdings[i]

    env.holdings = new_holdings
    env.fees_paid_total += rebalance_cost
    if turnover > 1e-4:
        env.trades_count += 1
    return rebalance_cost, 0.0


def apply_discrete_action(env, action):
    """Process discrete actions.

    Returns a tuple:
        (fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price)
    Side‑effects on ``env`` (cash, holdings, logs, penalties) are performed
    inside this function. ``step_penalty`` and ``last_remap_note`` are stored
    on the environment instance for the caller to use.
    """
    action_type = action[0]
    asset_idx = action[1]
    amount_pct = float(action[2]) / 100.0
    amount_pct = np.clip(amount_pct, 0.0, 1.0)
    # Initialise locals
    fee_paid = 0.0
    realised_pnl = 0.0
    is_valid_sell = False
    trade_units = 0.0
    trade_price = 0.0
    step_penalty = 0.0
    env.last_remap_note = None

    if action_type == 0:
        amount_pct = 0.0

    if action_type == 1:  # BUY
        if amount_pct == 0.0:
            step_penalty += env.empty_buy_penalty
            action_type = 0
            env.last_remap_note = "empty BUY remapped to HOLD"
        elif asset_idx < env.num_assets:
            if env.cash <= 1e-9:
                step_penalty += env.illegal_buy_penalty
                action_type = 0
                env.last_remap_note = f"illegal action (BUY, {env.asset_names[asset_idx]}, {amount_pct * 100:.0f}%): no cash, remapped to HOLD"
                amount_pct = 0.0
            else:
                buy_amount_usd = env.cash * amount_pct
                if buy_amount_usd > 0:
                    trade_price = env.prices_arr[env.current_step - 1][asset_idx]
                    fee_paid = buy_amount_usd * env.fee_rate
                    amount_after_fee = buy_amount_usd - fee_paid
                    trade_units = amount_after_fee / trade_price
                    env.cash -= buy_amount_usd
                    env.holdings[asset_idx] += trade_units
                    env.fees_paid_total += fee_paid
                    env.trades_count += 1
                    env.total_cost_basis[asset_idx] += amount_after_fee
                    env.avg_entry_price[asset_idx] = (
                        env.total_cost_basis[asset_idx] / env.holdings[asset_idx]
                        if env.holdings[asset_idx] > 0
                        else 0.0
                    )
    elif action_type == 2:  # SELL
        if amount_pct == 0.0:
            step_penalty += env.empty_sell_penalty
            action_type = 0
            env.last_remap_note = "empty SELL remapped to HOLD"
        elif asset_idx < env.num_assets:
            if env.holdings[asset_idx] > 0:
                units_to_sell = env.holdings[asset_idx] * amount_pct
                if units_to_sell > 0:
                    is_valid_sell = True
                    trade_price = env.prices_arr[env.current_step - 1][asset_idx]
                    trade_units = units_to_sell
                    gross_proceeds = units_to_sell * trade_price
                    fee_paid = gross_proceeds * env.fee_rate
                    proceeds = gross_proceeds - fee_paid
                    env.cash += proceeds
                    env.holdings[asset_idx] -= trade_units
                    env.fees_paid_total += fee_paid
                    env.trades_count += 1
                    if env.holdings[asset_idx] <= 1e-9:
                        env.avg_entry_price[asset_idx] = 0.0
                        env.total_cost_basis[asset_idx] = 0.0
                        # Add this to physically destroy the micro-dust
                        env.holdings[asset_idx] = 0.0
                    else:
                        env.total_cost_basis[asset_idx] *= env.holdings[asset_idx] / (
                            env.holdings[asset_idx] + units_to_sell
                        )
                    realised_pnl = proceeds - (
                        trade_units * env.avg_entry_price[asset_idx]
                    )
            else:
                step_penalty += env.illegal_sell_penalty
                action_type = 0
                env.last_remap_note = f"illegal action (SELL, {env.asset_names[asset_idx]}, {amount_pct * 100:.0f}%) remapped to HOLD"
                amount_pct = 0.0
        else:
            action_type = 0
            amount_pct = 0.0
            env.last_remap_note = "invalid asset index remapped to HOLD"

    # Store penalty for caller
    env._step_penalty = step_penalty
    return fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price


def get_action_mask(env):
    """
    Action mask for MultiDiscrete([3, num_assets, 101])
    Returns a flat boolean array of size (3 + num_assets + 101).
    """
    # 1. Action Type Mask (Size: 3) -> [HOLD, BUY, SELL]
    action_type_mask = np.ones(3, dtype=bool)

    # Block BUY globally if cash is exhausted
    if env.cash <= 1e-8:
        action_type_mask[1] = False

    # Block SELL globally if we hold absolutely no assets
    if np.sum(env.holdings) <= 1e-8:
        action_type_mask[2] = False

    # 2. Asset Index Mask (Size: num_assets)
    # All assets MUST remain True. If we mask an empty asset to prevent a SELL,
    # we accidentally prevent the agent from BUYING it.
    asset_mask = np.ones(env.num_assets, dtype=bool)

    # 3. Amount Mask (Size: 101)
    amount_mask = np.ones(101, dtype=bool)

    # Concatenate for sb3-contrib ActionMasker
    return np.concatenate([action_type_mask, asset_mask, amount_mask])
