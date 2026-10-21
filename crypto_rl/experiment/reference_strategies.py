"""Rule-based reference strategies (ROADMAP P2-2).

Each strategy picks the assets to hold from past prices only and maps them to
target-weight bins; the rest of the portfolio is cash.

The strategies act through the same env as the agent: the same action space,
fees, rebalance band, decision interval and evaluation window. If none of them
beats the passive baselines after fees, the agent is unlikely to either.

At a decision with ``env.current_step == t`` the trade executes at
``prices[t-1]``, the last closed bar, so a strategy reads ``prices[:t]``. Until
``lookback`` bars of history exist, it holds cash.
"""

import numpy as np


class RuleBasedStrategy:
    """Base class: a "model" that `_run_episode` can run in place of PPO.

    ``predict`` ignores the observation and reads the attached env's raw
    prices instead. Call ``attach`` after the env's reset.
    """

    def __init__(self, lookback: int):
        if lookback < 1:
            raise ValueError(f"lookback must be at least 1 bar, got {lookback}")
        self.lookback = lookback
        self.env = None
        # Invested fraction of PV at each decision, before it trades
        self.exposures: list[float] = []

    def attach(self, env) -> None:
        self.env = env
        self.exposures = []

    def select(self, prices: np.ndarray, last: int) -> np.ndarray:
        """Boolean mask of assets to hold, from ``prices[: last + 1]``."""
        raise NotImplementedError

    def held_bin(self, env) -> int:
        """Weight bin of a selected asset.

        The top bin by default: `target_weights` scales targets that sum to
        more than 1 down proportionally, so this is an equal weight across the
        selected assets, capped at ``max_asset_allocation`` each.
        """
        return len(env.weight_bins) - 1

    def predict(self, obs, deterministic: bool = True):
        env = self.env
        if env is None:
            raise RuntimeError("attach() the strategy to an env before predict()")
        last = env.current_step - 1
        if last - self.lookback < 0:
            selected = np.zeros(env.num_assets, dtype=bool)
        else:
            selected = self.select(env.prices_arr, last)
        self.exposures.append(1.0 - env.cash / env.portfolio_value)
        action = np.where(selected, self.held_bin(env), 0)
        # One row per env, as from a VecEnv policy
        return action[None, :], None


def _valid(prices: np.ndarray) -> np.ndarray:
    return np.isfinite(prices) & (prices > 0)


class TimeSeriesMomentum(RuleBasedStrategy):
    """Hold each asset whose return over the last ``lookback`` bars is positive.

    Each asset has a fixed sleeve, the largest bin up to ``1 / num_assets``
    (10% for 9 assets with 5% bins), so a signal flip trades only that asset.
    Equal weights across the selected assets would instead shift every other
    weight by more than the rebalance band at each flip.
    """

    def held_bin(self, env) -> int:
        return int(np.flatnonzero(env.weight_bins <= 1.0 / env.num_assets + 1e-9)[-1])

    def select(self, prices: np.ndarray, last: int) -> np.ndarray:
        now = np.asarray(prices[last], dtype=float)
        then = np.asarray(prices[last - self.lookback], dtype=float)
        return _valid(now) & _valid(then) & (now > then)


class BasketTrendFilter(RuleBasedStrategy):
    """Hold the equal-weight basket while it is above its moving average, else cash.

    The basket is the geometric mean of the asset prices, so each asset moves
    it by its log return, whatever its price level. The moving average covers
    the closes from ``last - lookback`` to ``last``, the same span as
    `TimeSeriesMomentum`. Assets without a valid price over the whole span are
    left out.
    """

    def select(self, prices: np.ndarray, last: int) -> np.ndarray:
        window = np.asarray(prices[last - self.lookback : last + 1], dtype=float)
        assets = _valid(window).all(axis=0)
        if not assets.any():
            return assets
        basket = np.exp(np.log(window[:, assets]).mean(axis=1))
        if basket[-1] > basket.mean():
            return assets
        return np.zeros_like(assets)


def default_strategies() -> dict[str, RuleBasedStrategy]:
    """The reference set reported next to the passive baselines (bars = minutes)."""
    return {
        "momentum_1h": TimeSeriesMomentum(lookback=60),
        "momentum_24h": TimeSeriesMomentum(lookback=1440),
        "basket_trend_1h": BasketTrendFilter(lookback=60),
        "basket_trend_24h": BasketTrendFilter(lookback=1440),
    }
