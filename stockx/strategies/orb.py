from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import opening_range


@register
class OpeningRangeBreakout(Strategy):
    name = "orb"
    display_name = "Opening Range Breakout"

    def default_params(self) -> dict:
        return {"opening_range_minutes": 15, "exit_mode": "level_reversal", "target_multiple": 1.0}

    def param_choices(self) -> dict:
        return {"exit_mode": ["level_reversal", "or_midpoint", "measured_move"]}

    def _entry_conditions(self, bars: pd.DataFrame):
        orange = opening_range(bars, self.params["opening_range_minutes"])
        entry_long = bars["close"] > orange["or_high"]
        entry_short = bars["close"] < orange["or_low"]
        return orange, entry_long, entry_short

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        orange, entry_long, entry_short = self._entry_conditions(bars)
        exit_mode = self.params.get("exit_mode", "level_reversal")

        if exit_mode in ("level_reversal", "measured_move"):
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "or_midpoint":
            # Exit at the opening range's own midpoint instead of waiting
            # for a full re-cross of the breakout level -- tighter, and
            # can't overlap with entry since the midpoint always sits
            # strictly inside the OR (below or_high, above or_low).
            midpoint = (orange["or_high"] + orange["or_low"]) / 2
            return hold_until_exit(entry_long, entry_short, bars["close"] <= midpoint, bars["close"] >= midpoint)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "measured_move":
            return None
        orange, entry_long, entry_short = self._entry_conditions(bars)
        or_range = orange["or_high"] - orange["or_low"]
        multiple = self.params["target_multiple"]
        # Classic ORB technique: project the opening range's own height as
        # the profit target beyond the breakout level, in the breakout's
        # direction.
        target_price = pd.Series(float("nan"), index=bars.index)
        target_price[entry_long] = (orange["or_high"] + multiple * or_range)[entry_long]
        target_price[entry_short] = (orange["or_low"] - multiple * or_range)[entry_short]
        return pd.DataFrame({"target_price": target_price})
