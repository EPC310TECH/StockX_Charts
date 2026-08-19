import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import bollinger_bands


@register
class BollingerReversion(Strategy):
    """Mean-reversion off the bands: long when price closes at/below the
    lower band (stretched below its own recent average, expecting a
    reversion), short at/above the upper band. Complements rsi_reversion.py
    (a different reversion signal -- price/volatility-based rather than a
    momentum oscillator) rather than duplicating it."""

    name = "bollinger_reversion"
    display_name = "Bollinger Band Reversion"

    def default_params(self) -> dict:
        return {"period": 20, "num_std": 2.0, "exit_mode": "threshold_cross"}

    def param_choices(self) -> dict:
        return {"exit_mode": ["threshold_cross", "midline", "opposite_extreme"]}

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        upper, mid, lower = bollinger_bands(bars["close"], self.params["period"], self.params["num_std"])
        close = bars["close"]
        entry_long = close <= lower
        entry_short = close >= upper

        exit_mode = self.params.get("exit_mode", "threshold_cross")
        if exit_mode == "threshold_cross":
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "midline":
            # Ride the full reversion to the middle band (its own SMA)
            # instead of bailing the instant price re-crosses the outer band.
            return hold_until_exit(entry_long, entry_short, close >= mid, close <= mid)
        if exit_mode == "opposite_extreme":
            return hold_until_exit(entry_long, entry_short, close >= upper, close <= lower)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
