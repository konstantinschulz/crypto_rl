import numpy as np


def build_observation(env) -> np.ndarray:
    """
    Constructs the observation vector on the fly to save memory.

    No lookahead: at env.current_step == t, the agent has only seen bars up to
    t-1 (the last closed bar). The step that consumes this observation fills
    at prices[t-1] and is rewarded on prices[t-1] -> prices[t], so every
    market-derived input here (windows, static features, volatility, portfolio
    marks) must come from index t-1 or earlier. Reading index t would expose
    the reward's end price to the policy.

    Vector structure (maintains exact backward compatibility for pre-trained weights):
    1. Macro features (5)
    2. Base window price changes (W * N)
    3. 1-min 30-bar price changes (30 * N)
    4. 5-min 24-bar price changes (24 * N)
    5. 60-min 24-bar price changes (24 * N)
    6. Statistical indicators (7 * N)
    7. Dynamic portfolio features (1 + 3 * N)
    """
    # Cap t at the maximum index to prevent out-of-bounds on the terminal step
    t = min(env.current_step, len(env.prices_arr) - 1)
    # Last closed bar: the newest data the agent may observe (see docstring)
    last = t - 1

    N = env.num_assets
    W = env.config.window_size
    prices = env.prices_arr

    last_price = prices[last]

    # Avoid division by zero when calculating relative changes
    safe_last = np.where(last_price > 0, last_price, 1e-8)
    # --- NEW: Extract last closed bar's volatility for normalization ---
    current_vol = env.norm_vol_arr[last]
    safe_vol = np.where(current_vol > 1e-8, current_vol, 1e-8)
    # ----------------------------------------------------------------

    def calc_rel_change(window_arr: np.ndarray) -> np.ndarray:
        """Fast vectorized relative change using broadcasting."""
        res = (window_arr / safe_last) - 1.0
        # If the reference price was <= 0, zero out that asset's history
        res[:, last_price <= 0] = 0.0
        # --- NEW: Normalize the price changes by the asset's current volatility ---
        # window_arr is shape (seq_len, N). safe_vol is shape (N,). 
        # Numpy broadcasting automatically divides per-asset.
        res = res / safe_vol
        # --------------------------------------------------------------------------
        return res.ravel()

    idx = 0

    # ------------------------------------------------------------------
    # 1. Macro features (stored at indices 0:5 in new precalc_static_obs)
    # ------------------------------------------------------------------
    env.obs_buf[idx : idx + env.macro_dim] = env.precalc_static_obs[
        last, 0 : env.macro_dim
    ]
    idx += env.macro_dim

    # ------------------------------------------------------------------
    # 2-5. Dynamic Time-Series Windows (Computed on the fly via fast views)
    # ------------------------------------------------------------------
    # Base window (W * N)
    env.obs_buf[idx : idx + W * N] = calc_rel_change(prices[t - W : t])
    idx += W * N

    # 1-min 30-bar window (30 * N)
    if t >= 30:
        env.obs_buf[idx : idx + 30 * N] = calc_rel_change(prices[t - 30 : t])
    else:
        env.obs_buf[idx : idx + 30 * N] = 0.0
    idx += 30 * N

    # 5-min 24-bar window (24 * N)
    if t >= 120:
        idxs_5 = np.arange(t - 120, t, 5)
        env.obs_buf[idx : idx + 24 * N] = calc_rel_change(prices[idxs_5])
    else:
        env.obs_buf[idx : idx + 24 * N] = 0.0
    idx += 24 * N

    # 60-min 24-bar window (24 * N)
    if t >= 1440:
        idxs_60 = np.arange(t - 1440, t, 60)
        env.obs_buf[idx : idx + 24 * N] = calc_rel_change(prices[idxs_60])
    else:
        env.obs_buf[idx : idx + 24 * N] = 0.0
    idx += 24 * N

    # ------------------------------------------------------------------
    # 6. Statistical & HTF indicators (stored after macro in precalc_static_obs)
    # ------------------------------------------------------------------
    ind_dim = env.static_per_asset_dim * N
    env.obs_buf[idx : idx + ind_dim] = env.precalc_static_obs[
        last, env.macro_dim : env.macro_dim + ind_dim
    ]
    idx += ind_dim

    # ------------------------------------------------------------------
    # 7. Dynamic Portfolio Features
    # ------------------------------------------------------------------
    safe_port_val = max(env.portfolio_value, 1e-8)

    # Cash percentage
    env.obs_buf[idx] = env.cash / safe_port_val
    idx += 1

    # Holdings percentage
    # Marked at the last closed bar, matching portfolio_value in step()
    current_prices = prices[last]
    asset_values = env.holdings * current_prices
    env.obs_buf[idx : idx + N] = asset_values / safe_port_val
    idx += N

    # Unrealised PnL percentage
    for i in range(N):
        if env.holdings[i] > 0 and env.avg_entry_price[i] > 0:
            env.unrealised_pnl_buf[i] = (
                current_prices[i] - env.avg_entry_price[i]
            ) / env.avg_entry_price[i]
        else:
            env.unrealised_pnl_buf[i] = 0.0

    env.obs_buf[idx : idx + N] = env.unrealised_pnl_buf
    idx += N

    # Has position flag
    has_pos = (env.holdings > 0).astype(np.float32)
    env.obs_buf[idx : idx + N] = has_pos
    idx += N

    current_drawdown = (env.portfolio_value - env.peak_portfolio_value) / max(
        env.peak_portfolio_value, 1e-8
    )
    env.obs_buf[idx] = current_drawdown
    idx += 1

    return env.obs_buf
