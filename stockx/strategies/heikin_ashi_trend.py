from typing import Optional

import numpy as np
import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import heikin_ashi


@register
class HeikinAshiTrend(Strategy):
    """Heikin-Ashi 'shaven candle' trend rule: long on a bullish HA candle
    with (near enough) no lower wick -- open sitting at the low, the
    classic strong-uptrend HA shape -- short on the bearish mirror image.
    A HA candle with wicks on both sides (indecision/consolidation) is
    flat. `wick_tolerance` is relative to that candle's own HA range
    rather than a fixed price, so it means the same thing across symbols
    and price levels."""

    name = "heikin_ashi_trend"
    display_name = "Heikin-Ashi Trend"

    def default_params(self) -> dict:
        return {"wick_tolerance": 0.05, "exit_mode": "shaven_candle"}

    def param_choices(self) -> dict:
        return {"exit_mode": ["shaven_candle", "any_reversal", "entry_candle_stop"]}

    def _entry_conditions(self, bars: pd.DataFrame):
        ha = heikin_ashi(bars)
        rng = (ha["high"] - ha["low"]).replace(0, np.nan)
        tol = self.params["wick_tolerance"]

        bullish = ha["close"] > ha["open"]
        bearish = ha["close"] < ha["open"]
        no_lower_wick = (ha["open"] - ha["low"]) <= tol * rng
        no_upper_wick = (ha["high"] - ha["open"]) <= tol * rng

        entry_long = (bullish & no_lower_wick).fillna(False)
        entry_short = (bearish & no_upper_wick).fillna(False)
        return ha, bullish, bearish, entry_long, entry_short

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        exit_mode = self.params.get("exit_mode", "shaven_candle")
        ha, bullish, bearish, entry_long, entry_short = self._entry_conditions(bars)

        if exit_mode in ("shaven_candle", "entry_candle_stop"):
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "any_reversal":
            # Hold through wicked (indecision) candles once trend-entered,
            # only exiting on an outright opposite-colored HA candle,
            # rather than requiring the exit candle to be shaven too.
            return hold_until_exit(entry_long, entry_short, bearish.fillna(False), bullish.fillna(False))
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "entry_candle_stop":
            return None
        ha, _bullish, _bearish, entry_long, entry_short = self._entry_conditions(bars)
        # The entry candle's own HA low/high as a fixed stop -- a
        # shaven-candle entry's own extreme is a natural, already-computed
        # invalidation level (price trading back through where the
        # signal candle started is a direct contradiction of the "clean
        # trend candle" premise), not read from a foreign indicator.
        stop_price = pd.Series(float("nan"), index=bars.index)
        stop_price[entry_long] = ha["low"][entry_long]
        stop_price[entry_short] = ha["high"][entry_short]
        return pd.DataFrame({"stop_price": stop_price})
