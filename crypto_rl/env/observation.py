import numpy as np

def build_observation(env) -> np.ndarray:
    """Construct observation vector for the environment.

    Mirrors the original `_get_obs` logic from MinimalCryptoEnv.
    """
    # Static observations precomputed
    env.obs_buf[: env.static_dim] = env.precalc_static_obs[env.current_step]

    current_prices = env.prices_arr[env.current_step - 1]
    safe_prices = np.nan_to_num(current_prices, nan=0.0, posinf=0.0, neginf=0.0)

    asset_value = np.sum(env.holdings * safe_prices)
    total_val = env.cash + asset_value

    idx = env.static_dim
    if total_val > 1e-8:
        env.obs_buf[idx] = env.cash / total_val
        idx += 1
        env.obs_buf[idx: idx + env.num_assets] = np.divide(
            env.holdings * safe_prices,
            total_val,
            out=np.zeros(env.num_assets, dtype=np.float32),
            where=total_val > 1e-8,
        )
    else:
        env.obs_buf[idx] = 0.0
        idx += 1
        env.obs_buf[idx: idx + env.num_assets] = 0.0

    idx += env.num_assets

    env.unrealised_pnl_buf.fill(0.0)
    mask = env.avg_entry_price > 1e-8
    if np.any(mask):
        env.unrealised_pnl_buf[mask] = (safe_prices[mask] - env.avg_entry_price[mask]) / env.avg_entry_price[mask]

    env.obs_buf[idx: idx + env.num_assets] = np.nan_to_num(
        env.unrealised_pnl_buf, nan=0.0, posinf=0.0, neginf=0.0
    )
    idx += env.num_assets

    has_position = (env.holdings > 1e-9).astype(np.float32)
    env.obs_buf[idx: idx + env.num_assets] = has_position

    np.nan_to_num(env.obs_buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return env.obs_buf
