from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, Optional, Type

import pandas as pd


class Signal(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


STRATEGY_REGISTRY: Dict[str, Type["Strategy"]] = {}


def register(cls: Type["Strategy"]) -> Type["Strategy"]:
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


class Strategy(ABC):
    """Base class for all day-trading strategies.

    Subclasses must set class attributes `name` (unique registry key) and
    `display_name` (human-readable), and implement `generate_signals`.
    """

    name: str = ""
    display_name: str = ""

    def __init__(self, **params):
        self.params = {**self.default_params(), **params}

    def default_params(self) -> dict:
        return {}

    def param_choices(self) -> Dict[str, list]:
        """Declares which default_params() entries are a fixed choice of
        strings (e.g. an exit-mode selector) rather than a free numeric
        value -- {param_name: [valid, choices]}. Empty by default. Purely
        a UI/coercion hint: the web dashboard's single-strategy backtest
        form renders a dropdown for any param name listed here instead of
        a numeric input, and web_backtest._coerce_params skips numeric
        coercion for it. Nothing else (the leaderboard/comparison view,
        walk-forward, the CLI) reads this -- they run every strategy via
        get_registered_strategies() with default_params() untouched, so a
        param declared here still needs a sensible default that preserves
        that strategy's original behavior.
        """
        return {}

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """Compute a desired-position signal for each bar.

        bars: single-symbol OHLCV DataFrame, sorted tz-aware DatetimeIndex,
        1-minute bars, columns ['open', 'high', 'low', 'close', 'volume'].
        May span multiple trading days.

        Returns a pd.Series[int] aligned to bars.index with values from
        Signal, decided using only data up to and including that bar (no
        lookahead) -- the backtest engine fills at the *next* bar's open.
        """
        raise NotImplementedError

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Optional per-bar stop-loss/take-profit price levels for a
        position entered on that bar.

        Returns None (the default) to trade with no bracket exit -- the
        engine then only closes a position on the next signal change, same
        as if this method didn't exist. Strategies that want a bracket
        exit return a DataFrame aligned to bars.index with 'stop_price'
        and/or 'target_price' columns (NaN in either where that bar has no
        level); the engine reads the level at a trade's *entry* bar only
        and holds it fixed for that trade's duration, then watches each
        bar's high/low for a touch.
        """
        return None
