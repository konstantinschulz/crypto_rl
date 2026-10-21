from __future__ import annotations  # avoid circular import issues with TYPE_CHECKING

from typing import TYPE_CHECKING, Any

import numpy as np

from crypto_rl.env.action_processing import rebalance, target_weights
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
    reward_components: dict[str, float],
) -> None:
    """
    Core accounting logic for closing positions.
    Calculates PnL, updates win rates, and applies the optional turnover penalty.
    """
    env.total_closed_trades += 1
    env.per_asset_trades[asset_idx] += 1

    revenue = amount_sold * trade_price * (1.0 - env.fee_rate)

    # Safely compute cost basis (entry drag is already baked into avg_entry_price)
    cost_basis = amount_sold * env.avg_entry_price[asset_idx]

    adjusted_pnl = revenue - cost_basis
    env.per_asset_realized_pnl[asset_idx] += adjusted_pnl

    if cost_basis > 1e-8:
        trade_weight = (
            cost_basis / prev_portfolio_value if prev_portfolio_value > 1e-8 else 0.0
        )

        # A win requires net revenue to exceed the exact cost we paid for those units
        if revenue > cost_basis:
            env.winning_trades_count += 1
            env.per_asset_wins[asset_idx] += 1

        # Optional turnover shaping
        holding_period = env.current_step - env.entry_step[asset_idx]
        if holding_period < env.config.turnover_penalty_steps_threshold:
            reward_components["turnover_penalty"] -= (
                env.config.turnover_penalty * trade_weight
            )


def step_env(
    env: MinimalCryptoEnv, action
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Execute one decision step for MinimalCryptoEnv (ROADMAP P2-1).

    The action rebalances once at ``prices[t-1]``, then the book is held for
    ``config.decision_interval`` one-minute bars, ``t`` to ``t+k-1`` (fewer
    if the episode ends first). PV, drawdown and the benchmark are tracked
    per bar, so the step reward is the sum of the per-bar rewards and
    ``portfolio_values`` keeps one mark per minute.

    Returns:
        obs: np.ndarray - The next observation after the step.
        reward: float - The reward obtained from the step.
        terminated: bool - Always False; the env has no terminal states.
        truncated: bool - True at the episode's time limit (``max_steps``).
        info: dict - Additional information about the step, including reward components.
    """
    prev_portfolio_value = env.portfolio_value
    current_prices = env.prices_arr[env.current_step - 1]
    # Bars held after this decision: t .. t+n_bars-1
    n_bars = max(1, min(env.config.decision_interval, env.max_steps - env.current_step))
    held_prices = env.prices_arr[env.current_step : env.current_step + n_bars]

    # --- SNAPSHOT HOLDINGS BEFORE ACTION ---
    old_holdings = np.copy(env.holdings)
    target = target_weights(env, action)
    fees, units = rebalance(env, target)

    # Reward decomposition; the step reward is the sum of these terms
    reward_components = {
        "log_return": 0.0,
        "benchmark": 0.0,
        "drawdown_penalty": 0.0,
        "turnover_penalty": 0.0,
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

    # Hold the book over the interval: no trades, so cash and holdings are fixed
    env.current_step += n_bars
    # The randomized episode boundary is a time limit, not a terminal state:
    # flag it as truncated so PPO bootstraps the value of the open book.
    truncated = env.current_step >= env.max_steps
    # Mark-to-market PV at every held bar. Open positions are never
    # force-liquidated.
    bar_pvs = env.cash + held_prices @ env.holdings.astype(np.float64)
    current_asset_value = float(bar_pvs[-1] - env.cash)
    env.portfolio_value = float(bar_pvs[-1])
    env.portfolio_values.extend(bar_pvs.tolist())

    info: dict[str, Any] = {}

    if truncated:
        # Reported final PV is the liquidation value: the exit fee is charged
        # here, as the passive baselines are charged, without executing a sell.
        # Trade counts and win rates therefore only reflect policy sells.
        exit_fees = float(current_asset_value * env.fee_rate)
        info["per_asset_stats"] = get_per_asset_summary(env)
        info["final_portfolio_value"] = env.portfolio_value - exit_fees
        info["final_exit_fees"] = exit_fees
        info["final_trades_count"] = env.trades_count
        info["final_fees_paid"] = env.fees_paid_total + exit_fees

        sum_per_asset_pnl = sum(
            s["total_pnl"] for s in info["per_asset_stats"].values()
        )
        actual_pnl = info["final_portfolio_value"] - env.config.budget_initial

        # Log warning if discrepancy exceeds $0.05
        if abs(sum_per_asset_pnl - actual_pnl) > 0.05:
            env.logger.debug(
                f"PnL Mismatch Detected! Portfolio PnL: ${actual_pnl:.2f} vs "
                f"Sum Per-Asset PnL: ${sum_per_asset_pnl:.2f}"
            )

    # ==========================================
    # REWARD
    # ==========================================
    # Mark-to-market: the per-step log returns sum to log(PV_T / PV_0), where
    # PV_T excludes the hypothetical exit fee. Fees paid are already inside PV.
    # Drawdown is tracked bar by bar, so a dip inside the interval that
    # recovers before its end still counts, as it would at 1-minute steps.
    peaks = np.maximum.accumulate(np.maximum(bar_pvs, env.peak_portfolio_value))
    env.peak_portfolio_value = float(peaks[-1])
    # Range: 0.0 (at peak) down to -1.0 (-100% loss)
    drawdowns = (bar_pvs - peaks) / peaks
    delta_drawdown = float(
        np.minimum(0.0, np.diff(drawdowns, prepend=env.previous_drawdown)).sum()
    )
    env.previous_drawdown = float(drawdowns[-1])

    reward_components["log_return"] = float(
        np.log(max(env.portfolio_value, 1e-8) / max(prev_portfolio_value, 1e-8))
    )
    if env.config.reward_type == "excess_log_return":
        # Unfloored log return of an equal-weight basket, rebalanced every
        # bar, summed over the held bars
        bar_prices = env.prices_arr[env.current_step - n_bars - 1 : env.current_step]
        gross = np.divide(
            bar_prices[1:],
            bar_prices[:-1],
            out=np.ones_like(bar_prices[1:]),
            where=bar_prices[:-1] > 1e-8,
        )
        reward_components["benchmark"] = -float(np.log(gross.mean(axis=1)).sum())
    reward_components["drawdown_penalty"] = (
        delta_drawdown * env.config.drawdown_penalty_coef
    )

    reward = sum(reward_components.values())

    if not env.config.disable_logging:
        log_action(
            env,
            env.current_step,
            target,
            units,
            fees,
            reward,
            prev_portfolio_value,
            current_prices,
            reward_components=reward_components,
        )

    info |= {
        "fees_paid": env.fees_paid_total,
        "trades_count": env.trades_count,
        "total_closed_trades": env.total_closed_trades,
        "winning_trades_count": env.winning_trades_count,
        "episode_count": env.episode_count,
        "reward_components": reward_components,
        "fill_counts": env.fill_counts,
        "traded_coins": env.traded_coins,
    }
    # Full history is only consumed from the terminal info. Including it every
    # step makes SB3's DummyVecEnv deepcopy an ever-growing list each step.
    if truncated:
        info["portfolio_values"] = env.portfolio_values

    return (
        env._get_obs(),
        float(reward),
        False,
        truncated,
        info,
    )
