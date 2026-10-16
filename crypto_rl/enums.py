from enum import Enum


class ActionType(Enum):
    """Enum for action types in the trading environment."""

    HOLD = 0
    BUY = 1
    SELL = 2
