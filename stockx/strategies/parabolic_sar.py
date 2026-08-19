from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.exit_utils import debounce_state
from stockx.strategies.indicators import parabolic_sar as compute_parabolic_sar


@register
class ParabolicSarStrategy(Strategy):
    name = "parabolic_sar"
    display_name = "Parabolic SAR"

    def default_params(self) -> dict:
        return {"af_step": 0.02, "af_max": 0.2, "exit_mode": "sar_flip", "confirm_bars": 2}

    def param_choices(self) -> dict:
        return {"exit_mode": ["sar_flip", "confirmed_flip", "sar_bracket"]}

    def _raw_state(self, bars: pd.DataFrame) -> pd.Series:
        sar = compute_parabolic_sar(bars, self.params["af_step"], self.params["af_max"])
        state = pd.Series(0, index=bars.index, dtype=int)
        state[bars["close"] > sar] = 1
        state[bars["close"] < sar] = -1
        return state

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        exit_mode = self.params.get("exit_mode", "sar_flip")
        raw_state = self._raw_state(bars)

        if exit_mode in ("sar_flip", "sar_bracket"):
            return raw_state
        if exit_mode == "confirmed_flip":
            # SAR is notoriously prone to whipsaw in choppy/ranging
            # markets -- require the flip to hold for confirm_bars before
            # acting on it, a well-known real-world refinement to raw SAR.
            return debounce_state(raw_state, self.params["confirm_bars"])
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.params.get("exit_mode") != "sar_bracket":
            return None
        # The SAR level itself, enforced as a hard intrabar stop from
        # entry -- protects against a gap or fast move against the
        # position before the next bar's signal would even register,
        # using the strategy's own core indicator rather than a foreign
        # reference level.
        sar = compute_parabolic_sar(bars, self.params["af_step"], self.params["af_max"])
        return pd.DataFrame({"stop_price": sar})
