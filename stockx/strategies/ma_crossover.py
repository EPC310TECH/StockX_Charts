from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import debounce_state
from stockx.strategies.indicators import ema


@register
class EmaCrossover(Strategy):
    name = "ma_crossover"
    display_name = "EMA 9/20 Crossover"

    def default_params(self) -> dict:
        return {"fast": 9, "slow": 20, "exit_mode": "signal_reversal", "confirm_bars": 3}

    def param_choices(self) -> dict:
        return {"exit_mode": ["signal_reversal", "confirmed_reversal", "slow_ema_stop"]}

    def _raw_state_per_day(self, bars: pd.DataFrame) -> pd.Series:
        fast_span = self.params["fast"]
        slow_span = self.params["slow"]

        def _per_day(close: pd.Series) -> pd.Series:
            fast_ema = ema(close, fast_span)
            slow_ema = ema(close, slow_span)
            state = pd.Series(0, index=close.index, dtype=int)
            state[fast_ema > slow_ema] = 1
            state[fast_ema < slow_ema] = -1
            return state

        day = bars.index.date
        state = bars["close"].groupby(day, sort=False, group_keys=False).apply(_per_day)
        return state.reindex(bars.index).fillna(0).astype(int)

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        exit_mode = self.params.get("exit_mode", "signal_reversal")
        raw_state = self._raw_state_per_day(bars)

        if exit_mode in ("signal_reversal", "slow_ema_stop"):
            # Both: original behavior -- the "stop" bracket in slow_ema_stop
            # mode is an *additional*, potentially-earlier exit trigger
            # layered on top via generate_stop_target below, not a change
            # to the signal itself.
            return raw_state
        if exit_mode == "confirmed_reversal":
            # Debounced per-day, same as the raw state, so a reversal on
            # day N's last bars can't bleed a confirmation window into
            # day N+1.
            day = bars.index.date
            confirm_bars = self.params["confirm_bars"]
            debounced = raw_state.groupby(day, sort=False, group_keys=False).apply(
                lambda s: debounce_state(s, confirm_bars)
            )
            return debounced.reindex(bars.index).fillna(0).astype(int)
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "slow_ema_stop":
            return None
        slow_span = self.params["slow"]

        def _per_day(close: pd.Series) -> pd.Series:
            return ema(close, slow_span)

        day = bars.index.date
        slow_ema = bars["close"].groupby(day, sort=False, group_keys=False).apply(_per_day)
        slow_ema = slow_ema.reindex(bars.index)
        # A single stop level, held fixed at whatever the slow EMA read at
        # entry -- the engine already checks stop_price against the
        # adverse side for whichever direction a run actually is (low, for
        # a long; high, for a short), so one column serves both sides
        # correctly without branching on position direction here. No
        # target_price -- this is a stop-only bracket, exit still comes
        # from the signal otherwise.
        return pd.DataFrame({"stop_price": slow_ema})
