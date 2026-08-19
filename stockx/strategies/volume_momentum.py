from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register


@register
class VolumeMomentumBreakout(Strategy):
    name = "volume_momentum"
    display_name = "Volume Momentum Breakout"

    def default_params(self) -> dict:
        return {
            "lookback": 20, "volume_multiplier": 2.0,
            "exit_mode": "level_reversal", "target_multiple": 1.0,
        }

    def param_choices(self) -> dict:
        return {"exit_mode": ["level_reversal", "swing_stop", "measured_move"]}

    def _levels_and_entries(self, bars: pd.DataFrame):
        lookback = self.params["lookback"]
        multiplier = self.params["volume_multiplier"]

        # Use prior `lookback` bars only (exclude current bar) to avoid
        # the current bar's own extreme value trivially satisfying the
        # breakout condition against itself.
        rolling_high = bars["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        rolling_low = bars["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        avg_volume = bars["volume"].shift(1).rolling(lookback, min_periods=lookback).mean()

        volume_spike = bars["volume"] > multiplier * avg_volume
        entry_long = ((bars["close"] > rolling_high) & volume_spike).fillna(False)
        entry_short = ((bars["close"] < rolling_low) & volume_spike).fillna(False)
        return rolling_high, rolling_low, entry_long, entry_short

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        _rolling_high, _rolling_low, entry_long, entry_short = self._levels_and_entries(bars)
        # Every exit_mode shares this exact signal -- the two non-default
        # modes only add a bracket exit (stop or target), they never
        # change when a position is entered or naturally exits.
        signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
        signal[entry_long] = Signal.LONG
        signal[entry_short] = Signal.SHORT
        return signal

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        exit_mode = self.params.get("exit_mode", "level_reversal")
        if exit_mode not in ("swing_stop", "measured_move"):
            return None
        rolling_high, rolling_low, entry_long, entry_short = self._levels_and_entries(bars)

        if exit_mode == "swing_stop":
            # Stop beyond the pre-breakout swing (the same rolling
            # low/high the breakout was measured against), rather than
            # waiting for price to fully retrace through the breakout
            # level -- and rather than requiring the volume spike itself
            # to persist, which the default mode implicitly does since a
            # spike is usually a 1-2 bar event.
            stop_price = pd.Series(float("nan"), index=bars.index)
            stop_price[entry_long] = rolling_low[entry_long]
            stop_price[entry_short] = rolling_high[entry_short]
            return pd.DataFrame({"stop_price": stop_price})

        # measured_move: project the pre-breakout range's own height as
        # the profit target beyond the breakout level.
        breakout_range = rolling_high - rolling_low
        multiple = self.params["target_multiple"]
        target_price = pd.Series(float("nan"), index=bars.index)
        target_price[entry_long] = (rolling_high + multiple * breakout_range)[entry_long]
        target_price[entry_short] = (rolling_low - multiple * breakout_range)[entry_short]
        return pd.DataFrame({"target_price": target_price})
