import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import rsi


@register
class RsiMeanReversion(Strategy):
    name = "rsi_reversion"
    display_name = "RSI Mean-Reversion"

    def default_params(self) -> dict:
        return {"period": 14, "oversold": 30, "overbought": 70, "exit_mode": "threshold_cross"}

    def param_choices(self) -> dict:
        return {"exit_mode": ["threshold_cross", "midline", "opposite_extreme"]}

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(bars["close"], self.params["period"])
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]
        entry_long = rsi_values < oversold
        entry_short = rsi_values > overbought

        exit_mode = self.params.get("exit_mode", "threshold_cross")
        if exit_mode == "threshold_cross":
            # Original behavior: stateless, exits the instant RSI ticks
            # back past the entry threshold.
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "midline":
            # Ride the full reversion back to neutral instead of bailing
            # at the first uptick out of the extreme zone.
            return hold_until_exit(entry_long, entry_short, rsi_values >= 50, rsi_values <= 50)
        if exit_mode == "opposite_extreme":
            # Ride through to the opposite extreme.
            return hold_until_exit(entry_long, entry_short, rsi_values >= overbought, rsi_values <= oversold)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
