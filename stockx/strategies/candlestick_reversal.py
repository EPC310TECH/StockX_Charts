from typing import Optional

import numpy as np
import pandas as pd

from stockx.strategies import candlestick_patterns as cp
from stockx.strategies.base import Signal, Strategy, register


@register
class CandlestickReversal(Strategy):
    """Named reversal-pattern half of the original candlestick_breakout
    strategy, split out to isolate its contribution -- see range_breakout.py
    for the other half (range contraction + breakout)."""

    name = "candlestick_reversal"
    display_name = "Candlestick Reversal Patterns"

    def default_params(self) -> dict:
        return {"exit_mode": "opposite_pattern", "fixed_hold_bars": 5, "target_r_multiple": 1.5}

    def param_choices(self) -> dict:
        return {"exit_mode": ["opposite_pattern", "fixed_hold", "candle_stop"]}

    def _entry_conditions(self, bars: pd.DataFrame):
        bullish_pattern = (
            cp.is_hammer(bars)
            | cp.is_bullish_engulfing(bars)
            | cp.is_morning_star(bars)
            | cp.is_three_white_soldiers(bars)
        ).fillna(False)
        bearish_pattern = (
            cp.is_hanging_man(bars)
            | cp.is_bearish_engulfing(bars)
            | cp.is_evening_star(bars)
            | cp.is_three_black_crows(bars)
        ).fillna(False)
        return bullish_pattern, bearish_pattern

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        entry_long, entry_short = self._entry_conditions(bars)
        exit_mode = self.params.get("exit_mode", "opposite_pattern")

        if exit_mode in ("opposite_pattern", "candle_stop"):
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "fixed_hold":
            # A candlestick reversal is meant to play out over a handful
            # of bars, not be held indefinitely until some opposite
            # pattern happens to fire -- exit unconditionally after
            # fixed_hold_bars, win or lose, unless an opposite-direction
            # pattern fires first.
            hold_bars = self.params["fixed_hold_bars"]
            el, es = entry_long.to_numpy(), entry_short.to_numpy()
            out = np.zeros(len(el), dtype=int)
            state = 0
            bars_left = 0
            for i in range(len(el)):
                if state != 0:
                    bars_left -= 1
                    if bars_left <= 0:
                        state = 0
                if state == 1 and es[i]:
                    state = 0
                elif state == -1 and el[i]:
                    state = 0
                if state == 0:
                    if el[i]:
                        state, bars_left = 1, hold_bars
                    elif es[i]:
                        state, bars_left = -1, hold_bars
                out[i] = state
            return pd.Series(out, index=bars.index, dtype=int)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "candle_stop":
            return None
        entry_long, entry_short = self._entry_conditions(bars)
        r_multiple = self.params["target_r_multiple"]

        # Stop beyond the triggering candle's own high/low -- the classic
        # candlestick-trading stop placement, using the same candle the
        # pattern itself fired on. Target is a fixed multiple of that same
        # risk distance, projected the other way.
        stop_price = pd.Series(float("nan"), index=bars.index)
        target_price = pd.Series(float("nan"), index=bars.index)

        entry_price = bars["close"]
        risk_long = (entry_price - bars["low"])[entry_long]
        stop_price[entry_long] = bars["low"][entry_long]
        target_price[entry_long] = (entry_price[entry_long] + r_multiple * risk_long)

        risk_short = (bars["high"] - entry_price)[entry_short]
        stop_price[entry_short] = bars["high"][entry_short]
        target_price[entry_short] = (entry_price[entry_short] - r_multiple * risk_short)

        return pd.DataFrame({"stop_price": stop_price, "target_price": target_price})
