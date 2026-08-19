from typing import Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy, register
from stockx.strategies.chart_patterns import find_all_chart_patterns


@register
class ChartPatternBreakout(Strategy):
    name = "chart_pattern_breakout"
    display_name = "Chart Pattern Breakout"

    def default_params(self) -> dict:
        return {"max_hold_bars": 30, "exit_mode": "level_reversal", "target_multiple": 1.0}

    def param_choices(self) -> dict:
        return {"exit_mode": ["level_reversal", "vertex_stop", "measured_move"]}

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """LONG/SHORT on a confirmed chart-pattern breakout (Head &
        Shoulders, Double Top/Bottom, triangles, wedges, flags), held on
        every subsequent bar while price stays beyond the breakout level
        (a simple, auditable proxy for 'the breakout is still valid'
        rather than re-extrapolating each pattern's sloped trendline
        forward), capped at max_hold_bars. FLAT otherwise -- same
        shift-by-one-bar fill and EOD-flatten handling as every other
        strategy. Identical across every exit_mode -- the non-default
        modes only add a bracket exit via generate_stop_target below, they
        never change entries or the natural level-reversal/max-hold exit."""
        signal = pd.Series(Signal.FLAT, index=bars.index, dtype=int)
        max_hold = self.params["max_hold_bars"]
        n = len(bars)

        for occurrences in find_all_chart_patterns(bars).values():
            for occ in occurrences:
                if occ.breakout_time not in bars.index:
                    continue
                start_i = bars.index.get_loc(occ.breakout_time)
                direction_val = Signal.LONG if occ.direction == "bullish" else Signal.SHORT
                end_i = min(start_i + max_hold, n - 1)
                for i in range(start_i, end_i + 1):
                    close = bars["close"].iloc[i]
                    still_valid = close >= occ.breakout_price if occ.direction == "bullish" else close <= occ.breakout_price
                    if not still_valid:
                        break
                    signal.iloc[i] = direction_val

        return signal

    def generate_stop_target(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        exit_mode = self.params.get("exit_mode", "level_reversal")
        if exit_mode not in ("vertex_stop", "measured_move"):
            return None

        stop_price = pd.Series(float("nan"), index=bars.index)
        target_price = pd.Series(float("nan"), index=bars.index)
        multiple = self.params["target_multiple"]

        for occurrences in find_all_chart_patterns(bars).values():
            for occ in occurrences:
                if occ.breakout_time not in bars.index or not occ.vertices:
                    continue
                bullish = occ.direction == "bullish"
                if exit_mode == "vertex_stop":
                    # The pattern's most extreme adverse vertex (lowest
                    # for a bullish pattern, highest for bearish) --
                    # already-computed points from the pattern detector,
                    # rather than a foreign reference. Using the single
                    # *last* vertex instead was tried first and empirically
                    # found unreliable: which side of the breakout price
                    # the last vertex lands on varies by pattern shape (a
                    # triangle's last touch sits near the breakout level
                    # itself, not below it), so ~1 in 4 wound up on the
                    # wrong side of entry and could never trigger. The
                    # overall min/max across every vertex is guaranteed
                    # correctly-sided by construction, since the pattern's
                    # own definition requires breaking out beyond its own
                    # extremes.
                    prices = [v[1] for v in occ.vertices]
                    stop_price.loc[occ.breakout_time] = min(prices) if bullish else max(prices)
                else:
                    # measured_move: the pattern's own overall height
                    # (its vertices' price spread), projected from the
                    # breakout level -- the standard technical-analysis
                    # measured-move target for chart patterns.
                    prices = [v[1] for v in occ.vertices]
                    height = max(prices) - min(prices)
                    target_price.loc[occ.breakout_time] = (
                        occ.breakout_price + multiple * height if bullish
                        else occ.breakout_price - multiple * height
                    )

        return pd.DataFrame({"stop_price": stop_price, "target_price": target_price})
