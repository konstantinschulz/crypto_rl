"""Target-weight actions (ROADMAP P1-3).

The action holds one bin index per asset. Bin ``k`` is a target weight of
``k * weight_step`` of portfolio value, up to ``max_asset_allocation``. The
env rebalances toward the target at the last closed price, skipping assets
whose weight is already within ``rebalance_band`` of it. Every action is
valid, so there is no action mask and nothing to remap or penalize.
"""

import numpy as np


def weight_bins(config) -> np.ndarray:
    """Target weights the action space can select, from 0 to the per-asset cap."""
    n_steps = round(config.max_asset_allocation / config.weight_step)
    if n_steps < 1 or not np.isclose(
        n_steps * config.weight_step, config.max_asset_allocation
    ):
        raise ValueError(
            f"max_asset_allocation ({config.max_asset_allocation}) must be a "
            f"positive multiple of weight_step ({config.weight_step})"
        )
    return np.arange(n_steps + 1) * config.weight_step


def target_weights(env, action) -> np.ndarray:
    """Map bin indices to target weights.

    If the targets sum to more than 1, they are scaled down proportionally,
    so the agent can ask for "fully invested" without picking exact bins.
    The rest of the portfolio is cash.
    """
    weights = env.weight_bins[np.asarray(action, dtype=np.int64)]
    total = weights.sum()
    if total > 1.0:
        weights = weights / total
    return weights


def rebalance(env, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trade toward ``target`` weights at ``prices[current_step - 1]``.

    Sells execute first so their proceeds can fund the buys. An asset is
    traded only if its weight is off target by at least ``rebalance_band``,
    except that a target of 0 always closes the whole position. Buys are
    scaled down together when cash runs short, which happens when the targets
    sum to 1, since fees come out of cash.

    Fee model, as for the baselines: a buy of S cash gets S / ((1 + fee) *
    price) units, and a sell of u units returns u * price * (1 - fee).

    Returns per-asset fees paid and per-asset units traded (negative = sold).
    """
    prices = env.prices_arr[env.current_step - 1]
    tradable = np.isfinite(prices) & (prices > 0)
    safe_prices = np.where(tradable, prices, 0.0)
    fee = env.fee_rate

    values = env.holdings * safe_prices
    pv = env.cash + float(values.sum())
    fees = np.zeros(env.num_assets)
    units = np.zeros(env.num_assets)
    if pv <= 1e-8:
        return fees, units

    current = values / pv
    gap = target - current
    close_out = (target == 0.0) & (env.holdings > 0)
    trade = tradable & ((np.abs(gap) >= env.config.rebalance_band) | close_out)

    for i in np.flatnonzero(trade & (gap < 0)):
        sold = (
            float(env.holdings[i])
            if target[i] == 0.0
            else min(-gap[i] * pv / prices[i], float(env.holdings[i]))
        )
        gross = sold * prices[i]
        fees[i] = gross * fee
        units[i] = -sold
        env.cash += gross - fees[i]
        env.holdings[i] -= sold
        if target[i] == 0.0 or env.holdings[i] <= 1e-9:
            env.holdings[i] = 0.0  # no dust left behind
        env.fill_counts["SELL"] += 1

    buys = np.flatnonzero(trade & (gap > 0))
    spend = gap[buys] * pv * (1.0 + fee)
    if spend.sum() > env.cash:
        spend *= env.cash / spend.sum()
    for i, s in zip(buys, spend):
        if s <= 0:
            continue
        cost = s / (1.0 + fee)
        fees[i] = cost * fee
        units[i] = cost / prices[i]
        env.cash -= s
        env.holdings[i] += units[i]
        env.fill_counts["BUY"] += 1
    env.cash = max(env.cash, 0.0)  # float rounding after spending all cash

    filled = np.flatnonzero(units != 0)
    env.trades_count += len(filled)
    env.fees_paid_total += float(fees.sum())
    env.per_asset_fees += fees
    env.traded_coins.update(env.asset_names[i] for i in filled)
    return fees, units
