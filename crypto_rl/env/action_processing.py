import numpy as np


def apply_continuous_action(env, action):
    """Process continuous actions.

    Returns a tuple (fee_paid, trade_units). ``trade_units`` is set to 0
    because continuous actions rebalance the whole portfolio rather than
    tracking a single trade quantity.
    """
    exp_action = np.exp(action - np.max(action))
    target_weights = exp_action / np.sum(exp_action)

    # 1. Enforce Per-Asset Max Allocation Caps
    base_cap = getattr(env, "max_asset_allocation", 1.0)
    target_vol = getattr(env, "target_volatility", 0.02)

    if hasattr(env, "asset_volatility"):
        current_vols = env.asset_volatility[env.current_step - 1]
        # Inverse volatility weighting: higher volatility -> lower cap
        dynamic_caps = base_cap * (target_vol / (current_vols + 1e-8))
        # Clamp caps to avoid edge cases: floor at 1%, ceiling at base_cap
        dynamic_caps = np.clip(dynamic_caps, 0.01, base_cap)
    else:
        dynamic_caps = np.full(env.num_assets, base_cap)

    asset_weights = target_weights[1:]
    # Calculate how much weight exceeds the cap across all assets
    excess_weight = np.sum(np.maximum(asset_weights - dynamic_caps, 0.0))
    # Clip assets to the cap and sweep the excess back into cash
    asset_weights = np.minimum(asset_weights, dynamic_caps)
    target_weights[0] += excess_weight
    target_weights[1:] = asset_weights

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

    # State assignment
    env.holdings = np.divide(
        new_asset_allocations,
        current_prices,
        out=np.zeros_like(new_asset_allocations),
        where=current_prices > 1e-8,
    )
    env.fees_paid_total += rebalance_cost

    if turnover > 1e-4:
        env.trades_count += 1

    # NOTE: We no longer update `env.avg_entry_price` or `env.total_cost_basis` here.
    # Those metrics are fully handled by the `deltas` loop in `minimal_env.py` directly
    # observing the change in `env.holdings` to prevent double-accounting.
    return rebalance_cost, 0.0


def apply_discrete_action(env, action):
    """Process discrete actions.

    Returns a tuple:
        (fee_paid, realised_pnl, is_valid_sell, trade_units, trade_price)
    Side‑effects on ``env`` (cash, holdings, logs, penalties) are performed
    inside this function. ``step_penalty`` and ``last_remap_note`` are stored
    on the environment instance for the caller to use.
    """
    # Initialize/reset the penalty tracker for this step
    env._step_penalty = 0.0
    action_type = action[0]
    asset_idx = action[1]
    amount_pct = float(action[2]) / 100.0
    amount_pct = np.clip(amount_pct, 0.0, 1.0)

    # Initialise locals
    fee_paid = 0.0
    realised_pnl = 0.0
    is_valid_sell = False
    trade_units = 0.0
    trade_price = env.prices_arr[env.current_step - 1][asset_idx]
    step_penalty = 0.0
    env.last_remap_note = None

    # Calculate portfolio value for concentration caps
    current_prices = env.prices_arr[env.current_step - 1]
    safe_prices = np.nan_to_num(current_prices, nan=0.0, posinf=0.0, neginf=0.0)
    current_asset_values = env.holdings * safe_prices
    curr_portfolio_val = env.cash + np.sum(current_asset_values)

    # 1. Catch and penalize invalid Sells
    if action_type == 2 and env.holdings[asset_idx] < 1e-8:
        env._step_penalty += env.illegal_sell_penalty
        env.last_remap_note = (
            f"illegal action (SELL, {env.asset_names[asset_idx]}) remapped to HOLD"
        )
        action_type = 0  # Force to HOLD to prevent phantom transactions/short selling

    # 2. Catch and penalize invalid Buys
    elif action_type == 1 and env.cash < 1e-8:
        env._step_penalty += env.illegal_buy_penalty
        env.last_remap_note = (
            f"illegal action (BUY, {env.asset_names[asset_idx]}) remapped to HOLD"
        )
        action_type = 0  # Force to HOLD

    if action_type == 0:
        amount_pct = 0.0

    if action_type == 1:  # BUY
        if amount_pct == 0.0:
            step_penalty += env.empty_buy_penalty
            action_type = 0
            env.last_remap_note = "empty BUY remapped to HOLD"
        elif asset_idx < env.num_assets:
            # 2a. Enforce Per-Asset Max Allocation Caps (Volatility Scaled)
            base_cap = getattr(env, "max_asset_allocation", 1.0)
            target_vol = getattr(env, "target_volatility", 0.02)
            if hasattr(env, "asset_volatility"):
                current_vol = env.asset_volatility[env.current_step - 1][asset_idx]
                dynamic_cap_pct = base_cap * (target_vol / (current_vol + 1e-8))
                dynamic_cap_pct = np.clip(dynamic_cap_pct, 0.01, base_cap)
            else:
                dynamic_cap_pct = base_cap

            max_cap_usd = curr_portfolio_val * dynamic_cap_pct
            current_exposure_usd = env.holdings[asset_idx] * trade_price
            room_to_buy_usd = max(0.0, max_cap_usd - current_exposure_usd)

            # 2b. Enforce Single-Step Spend Cap
            max_step_pct = getattr(env, "max_single_step_allocation", 1.0)
            max_step_spend = env.cash * max_step_pct

            # Bound the target spend by requested pct, single-step limits, and overall asset exposure limits
            target_spend_usd = min(
                env.cash * amount_pct, max_step_spend, room_to_buy_usd
            )

            if target_spend_usd > 0:
                # 3. Derive exact units and fee from the constrained target spend
                fee_multiplier = 1.0 + env.fee_rate
                cost_before_fee = target_spend_usd / fee_multiplier
                fee_paid = cost_before_fee * env.fee_rate
                trade_units = cost_before_fee / trade_price

                # 4. Deduct EXACT cash spent (no leakage)
                env.cash -= target_spend_usd  # target_spend_usd precisely equals (cost_before_fee + fee_paid)
                env.holdings[asset_idx] += trade_units
                env.fees_paid_total += fee_paid
                env.trades_count += 1

                # Note: `avg_entry_price` and `total_cost_basis` are intentionally left out here.
                # `minimal_env.py` manages cost basis tracking by observing the resulting `delta`.

    elif action_type == 2:  # SELL
        if amount_pct == 0.0:
            step_penalty += env.empty_sell_penalty
            action_type = 0
            env.last_remap_note = "empty SELL remapped to HOLD"
        elif asset_idx < env.num_assets:
            units_to_sell = env.holdings[asset_idx] * amount_pct
            if units_to_sell > 0:
                is_valid_sell = True
                trade_units = units_to_sell
                gross_proceeds = units_to_sell * trade_price
                fee_paid = gross_proceeds * env.fee_rate
                proceeds = gross_proceeds - fee_paid

                env.cash += proceeds
                env.holdings[asset_idx] -= trade_units
                env.fees_paid_total += fee_paid
                env.trades_count += 1

                # Calculate internal logging PnL. (Official PnL routing is managed in minimal_env.py)
                realised_pnl = proceeds - (trade_units * env.avg_entry_price[asset_idx])

                if env.holdings[asset_idx] <= 1e-9:
                    # Physically destroy the micro-dust
                    env.holdings[asset_idx] = 0.0

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
    # 1. Action Type Mask [Hold, Buy, Sell] (Length: 3)
    # Buy is valid if we have cash. Sell is valid if we hold ANY asset.
    can_buy = env.cash > 1e-8
    can_sell = np.any(env.holdings > 1e-8)
    mask_type = [True, can_buy, can_sell]

    # 2. Asset Index Mask (Length: env.num_assets)
    # All assets must remain True. SB3 evaluates dimensions independently.
    # If you masked out an unheld asset to prevent an invalid Sell,
    # you would simultaneously prevent the agent from Buying it.
    asset_mask = np.ones(env.num_assets, dtype=bool)

    # 3. Trade Amount Mask (Length: 101)
    # All percentages (0-100) are valid selections.
    amount_mask = np.ones(101, dtype=bool)

    # Concatenate into the final 1D array required by MaskablePPO
    return np.concatenate([mask_type, asset_mask, amount_mask], dtype=bool)
