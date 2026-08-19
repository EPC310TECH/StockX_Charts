from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import hold_until_exit


@register
class DualThrust(Strategy):
    """Opening-range breakout using a volatility-derived range instead of a
    fixed opening-minutes window (compare orb.py): range = max(HH-LC, HC-LL)
    over the `lookback_days` sessions strictly *before* today, then buy/sell
    triggers are today's open +/- k*range. Separate k1 (buy side) and k2
    (sell side) multipliers let the breakout be asymmetric."""

    name = "dual_thrust"
    display_name = "Dual Thrust"

    def default_params(self) -> dict:
        return {
            "lookback_days": 4, "k1": 0.5, "k2": 0.5,
            "exit_mode": "level_reversal", "target_multiple": 1.0,
        }

    def param_choices(self) -> dict:
        return {"exit_mode": ["level_reversal", "session_close_only", "measured_move"]}

    def _triggers_and_range(self, bars: pd.DataFrame):
        lookback = self.params["lookback_days"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]

        day = bars.index.date
        daily_high = bars["high"].groupby(day, sort=False).max()
        daily_low = bars["low"].groupby(day, sort=False).min()
        daily_close = bars["close"].groupby(day, sort=False).last()
        daily_open = bars["open"].groupby(day, sort=False).first()

        hh = daily_high.shift(1).rolling(lookback).max()
        lc = daily_close.shift(1).rolling(lookback).min()
        hc = daily_close.shift(1).rolling(lookback).max()
        ll = daily_low.shift(1).rolling(lookback).min()
        day_range = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

        buy_trigger = pd.Series((daily_open + k1 * day_range).reindex(day).to_numpy(), index=bars.index)
        sell_trigger = pd.Series((daily_open - k2 * day_range).reindex(day).to_numpy(), index=bars.index)
        day_range_per_bar = pd.Series(day_range.reindex(day).to_numpy(), index=bars.index)
        return buy_trigger, sell_trigger, day_range_per_bar

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        buy_trigger, sell_trigger, _ = self._triggers_and_range(bars)
        entry_long = bars["close"] > buy_trigger
        entry_short = bars["close"] < sell_trigger

        exit_mode = self.params.get("exit_mode", "level_reversal")
        if exit_mode in ("level_reversal", "measured_move"):
            signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
            signal[entry_long] = Signal.LONG
            signal[entry_short] = Signal.SHORT
            return signal
        if exit_mode == "session_close_only":
            # Hold from the first trigger through the rest of the session
            # regardless of later intraday level reversals -- the engine's
            # own EOD-flatten (or an opposite trigger) is the only exit,
            # rather than bailing the instant price dips back through the
            # trigger.
            day = bars.index.date
            no_exit = pd.Series(False, index=bars.index)
            held = (
                pd.DataFrame({"el": entry_long, "es": entry_short, "xl": no_exit, "xs": no_exit})
                .groupby(day, sort=False, group_keys=False)
                .apply(lambda g: hold_until_exit(g["el"], g["es"], g["xl"], g["xs"]))
            )
            return held.reindex(bars.index).fillna(0).astype(int)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "measured_move":
            return None
        buy_trigger, sell_trigger, day_range = self._triggers_and_range(bars)
        entry_long = bars["close"] > buy_trigger
        entry_short = bars["close"] < sell_trigger
        multiple = self.params["target_multiple"]

        target_price = pd.Series(float("nan"), index=bars.index)
        target_price[entry_long] = (buy_trigger + multiple * day_range)[entry_long]
        target_price[entry_short] = (sell_trigger - multiple * day_range)[entry_short]
        return pd.DataFrame({"target_price": target_price})
