from typing import Optional

import pandas as pd

from stockx.strategies import candlestick_patterns as cp
from stockx.strategies.base import Signal, Strategy, register


@register
class RangeBreakout(Strategy):
    """Contraction-breakout half of the original candlestick_breakout
    strategy, split out to isolate its contribution -- see
    candlestick_reversal.py for the other half (named reversal patterns)."""

    name = "range_breakout"
    display_name = "Range Contraction Breakout"

    def default_params(self) -> dict:
        return {
            "contraction_lookback": 20, "contraction_quantile": 0.2, "breakout_lookback": 20,
            "exit_mode": "level_reversal", "target_multiple": 1.0,
        }

    def param_choices(self) -> dict:
        return {"exit_mode": ["level_reversal", "contraction_stop", "measured_move"]}

    def _levels_and_entries(self, bars: pd.DataFrame):
        contraction = cp.range_contraction(
            bars, self.params["contraction_lookback"], self.params["contraction_quantile"]
        )
        was_contracting = contraction.shift(1, fill_value=False)
        rolling_high = bars["high"].shift(1).rolling(self.params["breakout_lookback"]).max()
        rolling_low = bars["low"].shift(1).rolling(self.params["breakout_lookback"]).min()
        entry_long = (was_contracting & (bars["close"] > rolling_high)).fillna(False)
        entry_short = (was_contracting & (bars["close"] < rolling_low)).fillna(False)
        return rolling_high, rolling_low, entry_long, entry_short

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        _rolling_high, _rolling_low, entry_long, entry_short = self._levels_and_entries(bars)
        signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
        signal[entry_long] = Signal.LONG
        signal[entry_short] = Signal.SHORT
        return signal

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        exit_mode = self.params.get("exit_mode", "level_reversal")
        if exit_mode not in ("contraction_stop", "measured_move"):
            return None
        rolling_high, rolling_low, entry_long, entry_short = self._levels_and_entries(bars)

        if exit_mode == "contraction_stop":
            # Stop at the far side of the pre-breakout contracted range --
            # a real retracement all the way back through the coil, not
            # just a re-cross of the breakout level itself.
            stop_price = pd.Series(float("nan"), index=bars.index)
            stop_price[entry_long] = rolling_low[entry_long]
            stop_price[entry_short] = rolling_high[entry_short]
            return pd.DataFrame({"stop_price": stop_price})

        # measured_move: the coiled range's own height, projected from the
        # breakout level -- classic contraction-breakout target sizing.
        coil_range = rolling_high - rolling_low
        multiple = self.params["target_multiple"]
        target_price = pd.Series(float("nan"), index=bars.index)
        target_price[entry_long] = (rolling_high + multiple * coil_range)[entry_long]
        target_price[entry_short] = (rolling_low - multiple * coil_range)[entry_short]
        return pd.DataFrame({"target_price": target_price})
