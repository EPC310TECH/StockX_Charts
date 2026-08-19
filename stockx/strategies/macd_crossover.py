import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit
from stockx.strategies.indicators import macd


@register
class MacdCrossover(Strategy):
    """MACD line vs its own signal line (EMA of the MACD line) -- distinct
    from ma_crossover.py's raw price EMA crossover, since it's a
    second-order comparison (momentum-of-momentum), not price itself."""

    name = "macd_crossover"
    display_name = "MACD Crossover"

    def default_params(self) -> dict:
        return {"fast": 12, "slow": 26, "signal": 9, "exit_mode": "signal_reversal"}

    def param_choices(self) -> dict:
        return {"exit_mode": ["signal_reversal", "histogram_fade", "zero_cross"]}

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, histogram = macd(
            bars["close"], self.params["fast"], self.params["slow"], self.params["signal"]
        )
        entry_long = macd_line > signal_line
        entry_short = macd_line < signal_line

        exit_mode = self.params.get("exit_mode", "signal_reversal")
        if exit_mode == "signal_reversal":
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal

        # The remaining modes exit *before* the signal line itself would
        # reverse, so entry has to be the crossover *moment* (an edge),
        # not "macd is still above/below signal" (a level) -- otherwise
        # that level condition is still true immediately after an early
        # exit, and hold_until_exit's same-bar re-entry check silently
        # undoes the exit before it's ever visible in the output. (Caught
        # this exact bug empirically: histogram_fade was producing output
        # byte-identical to signal_reversal until this fix.)
        crossed_up = (entry_long & ~entry_long.shift(1).fillna(False)).fillna(False)
        crossed_down = (entry_short & ~entry_short.shift(1).fillna(False)).fillna(False)

        if exit_mode == "histogram_fade":
            # Exit at the first sign momentum is fading (histogram
            # shrinking) rather than waiting for the full signal-line
            # cross -- tighter, still built from the same MACD histogram
            # the strategy is already named for.
            declining = (histogram.diff() < 0).fillna(False)
            rising = (histogram.diff() > 0).fillna(False)
            return hold_until_exit(crossed_up, crossed_down, declining, rising)
        if exit_mode == "zero_cross":
            # Exit on the MACD line's own zero-cross -- typically a later,
            # less sensitive signal than the signal-line cross.
            return hold_until_exit(crossed_up, crossed_down, (macd_line <= 0).fillna(False), (macd_line >= 0).fillna(False))
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
