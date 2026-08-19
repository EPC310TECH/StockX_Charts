from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import debounce_state
from stockx.strategies.indicators import awesome_oscillator


@register
class AwesomeOscillatorStrategy(Strategy):
    name = "awesome_oscillator"
    display_name = "Awesome Oscillator"

    def default_params(self) -> dict:
        return {
            "fast": 5, "slow": 34,
            "exit_mode": "zero_cross", "confirm_bars": 3, "swing_lookback": 10,
        }

    def param_choices(self) -> dict:
        return {"exit_mode": ["zero_cross", "confirmed_reversal", "swing_stop"]}

    def _raw_state(self, bars: pd.DataFrame) -> pd.Series:
        ao = awesome_oscillator(bars, self.params["fast"], self.params["slow"])
        state = pd.Series(0, index=bars.index, dtype=int)
        state[ao > 0] = 1
        state[ao < 0] = -1
        return state

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        exit_mode = self.params.get("exit_mode", "zero_cross")
        raw_state = self._raw_state(bars)

        if exit_mode in ("zero_cross", "swing_stop"):
            return raw_state
        if exit_mode == "confirmed_reversal":
            return debounce_state(raw_state, self.params["confirm_bars"])
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "swing_stop":
            return None
        lookback = self.params["swing_lookback"]
        # A simple rolling low/high as a "recent swing" proxy, held fixed
        # from entry -- the same recent-extreme idea a discretionary
        # trader means by "stop beyond the last swing," without pulling in
        # the full zigzag swing-point detector for a single bracket level.
        swing_low = bars["low"].shift(1).rolling(lookback, min_periods=1).min()
        swing_high = bars["high"].shift(1).rolling(lookback, min_periods=1).max()
        stop_price = pd.Series(float("nan"), index=bars.index)
        raw_state = self._raw_state(bars)
        stop_price[raw_state == 1] = swing_low[raw_state == 1]
        stop_price[raw_state == -1] = swing_high[raw_state == -1]
        return pd.DataFrame({"stop_price": stop_price})
