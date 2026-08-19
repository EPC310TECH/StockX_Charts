import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import session_vwap


@register
class VwapMeanReversion(Strategy):
    name = "vwap_reversion"
    display_name = "VWAP Mean-Reversion"

    def default_params(self) -> dict:
        return {"num_std": 1.0, "std_window": 30, "exit_mode": "threshold_cross"}

    def param_choices(self) -> dict:
        return {"exit_mode": ["threshold_cross", "midline", "opposite_extreme"]}

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        vwap = session_vwap(bars)
        deviation = bars["close"] - vwap
        rolling_std = deviation.rolling(
            self.params["std_window"], min_periods=self.params["std_window"]
        ).std()
        band = self.params["num_std"] * rolling_std

        entry_long = (deviation < -band).fillna(False)
        entry_short = (deviation > band).fillna(False)

        exit_mode = self.params.get("exit_mode", "threshold_cross")
        if exit_mode == "threshold_cross":
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "midline":
            # Ride the full reversion back to VWAP itself, not just out of
            # the band.
            return hold_until_exit(entry_long, entry_short, (deviation >= 0).fillna(False), (deviation <= 0).fillna(False))
        if exit_mode == "opposite_extreme":
            # Ride through to the opposite band.
            return hold_until_exit(entry_long, entry_short, (deviation > band).fillna(False), (deviation < -band).fillna(False))
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
