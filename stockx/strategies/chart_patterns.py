from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stockx.strategies.indicators import atr_pct

# Multi-bar chart patterns (Head & Shoulders, Double Top/Bottom, triangles,
# wedges, flags) -- geometrically different from candlestick_patterns.py's
# fixed 1-3 bar OHLC-ratio rules. These are defined by a sequence of swing
# highs/lows and trendlines spanning many bars, detected via a shared
# zigzag/swing-point pipeline rather than per-bar vectorized comparisons.


@dataclass
class SwingPoint:
    time: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


@dataclass
class ChartPatternOccurrence:
    name: str
    direction: str  # "bullish" | "bearish"
    vertices: List[Tuple[pd.Timestamp, float]]
    trendlines: List[Tuple[Tuple[pd.Timestamp, float], Tuple[pd.Timestamp, float]]]
    breakout_time: pd.Timestamp
    breakout_price: float
    confidence: float


def find_swing_points(bars: pd.DataFrame, threshold_pct: Optional[float] = None) -> List[SwingPoint]:
    """Zigzag-style pivots: track the running price extreme since the last
    confirmed pivot; a new pivot confirms once price reverses by at least
    `threshold_pct` from that extreme. Pivots always alternate high/low by
    construction -- the simplified 'skeleton' every chart-pattern detector
    below matches against, instead of noisy raw bars.

    `threshold_pct` defaults to a symbol-relative value (3x this bars
    series' own median ATR%, floored at 0.5%) rather than one fixed number
    for every symbol/interval, matching how range_contraction() elsewhere
    in this project scales off the symbol's own volatility instead of an
    absolute threshold.
    """
    if len(bars) < 3:
        return []

    if threshold_pct is None:
        atr_series = atr_pct(bars).dropna()
        base = float(atr_series.median()) * 100 if len(atr_series) else 1.0
        threshold_pct = max(base * 3, 0.5)

    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    times = bars.index
    n = len(bars)
    swings: List[SwingPoint] = []

    looking_for: Optional[str] = None  # None = not yet committed to a direction
    cand_high_i, cand_high_p = 0, highs[0]
    cand_low_i, cand_low_p = 0, lows[0]

    for i in range(1, n):
        if looking_for in (None, "high"):
            if highs[i] > cand_high_p:
                cand_high_p, cand_high_i = highs[i], i
            elif lows[i] <= cand_high_p * (1 - threshold_pct / 100):
                swings.append(SwingPoint(times[cand_high_i], float(cand_high_p), "high"))
                looking_for = "low"
                cand_low_p, cand_low_i = lows[i], i
                continue
        if looking_for in (None, "low"):
            if lows[i] < cand_low_p:
                cand_low_p, cand_low_i = lows[i], i
            elif highs[i] >= cand_low_p * (1 + threshold_pct / 100):
                swings.append(SwingPoint(times[cand_low_i], float(cand_low_p), "low"))
                looking_for = "high"
                cand_high_p, cand_high_i = highs[i], i
                continue

    # The extreme still being tracked when the data runs out never gets a
    # chance to confirm (that needs a subsequent threshold_pct reversal
    # that hasn't happened yet) so the loop above never appends it -- which
    # silently drops the most recent, most actionable swing point along
    # with it, and with it any pattern whose final vertex sits in that
    # trailing window. Append it as a final, still-forming pivot; like any
    # 'forming' read elsewhere in this project, it can shift if the caller
    # re-runs this against more bars later. Only applies once at least one
    # pivot has already confirmed (looking_for is "high"/"low", not None)
    # so the alternating high/low invariant above still holds.
    if looking_for == "high":
        swings.append(SwingPoint(times[cand_high_i], float(cand_high_p), "high"))
    elif looking_for == "low":
        swings.append(SwingPoint(times[cand_low_i], float(cand_low_p), "low"))

    return swings


def _line_value_at(p1: Tuple[pd.Timestamp, float], p2: Tuple[pd.Timestamp, float], t: pd.Timestamp) -> float:
    t1, y1 = p1
    t2, y2 = p2
    if t2 == t1:
        return y2
    frac = (t - t1) / (t2 - t1)
    return y1 + frac * (y2 - y1)


def _find_breakout_below_line(bars, after_time, p1, p2, max_bars_ahead=60):
    after_bars = bars[bars.index > after_time]
    for count, (ts, row) in enumerate(after_bars.iterrows(), start=1):
        if count > max_bars_ahead:
            break
        if row["close"] < _line_value_at(p1, p2, ts):
            return (ts, float(row["close"]))
    return None


def _find_breakout_above_line(bars, after_time, p1, p2, max_bars_ahead=60):
    after_bars = bars[bars.index > after_time]
    for count, (ts, row) in enumerate(after_bars.iterrows(), start=1):
        if count > max_bars_ahead:
            break
        if row["close"] > _line_value_at(p1, p2, ts):
            return (ts, float(row["close"]))
    return None


def _find_breakout_below_level(bars, after_time, level, max_bars_ahead=60):
    after_bars = bars[bars.index > after_time]
    for count, (ts, row) in enumerate(after_bars.iterrows(), start=1):
        if count > max_bars_ahead:
            break
        if row["close"] < level:
            return (ts, float(row["close"]))
    return None


def _find_breakout_above_level(bars, after_time, level, max_bars_ahead=60):
    after_bars = bars[bars.index > after_time]
    for count, (ts, row) in enumerate(after_bars.iterrows(), start=1):
        if count > max_bars_ahead:
            break
        if row["close"] > level:
            return (ts, float(row["close"]))
    return None


def find_head_and_shoulders(bars: pd.DataFrame, swings: List[SwingPoint]) -> List[ChartPatternOccurrence]:
    occurrences: List[ChartPatternOccurrence] = []
    for i in range(len(swings) - 4):
        window = swings[i:i + 5]
        if [s.kind for s in window] != ["high", "low", "high", "low", "high"]:
            continue
        left, trough1, head, trough2, right = window

        if not (head.price > left.price and head.price > right.price):
            continue
        pattern_height = head.price - min(trough1.price, trough2.price)
        if pattern_height <= 0:
            continue
        shoulder_diff = abs(left.price - right.price)
        if shoulder_diff > 0.5 * pattern_height:
            continue
        if (head.price - max(left.price, right.price)) < 0.1 * pattern_height:
            continue

        neckline = ((trough1.time, trough1.price), (trough2.time, trough2.price))
        breakout = _find_breakout_below_line(bars, right.time, *neckline)
        if breakout is None:
            continue

        symmetry_score = max(0.0, 1 - shoulder_diff / (0.5 * pattern_height))
        prominence = (head.price - max(left.price, right.price)) / pattern_height
        prominence_score = min(1.0, prominence / 0.3)
        confidence = round(100 * (0.5 * symmetry_score + 0.5 * prominence_score), 1)

        occurrences.append(ChartPatternOccurrence(
            name="head_and_shoulders", direction="bearish",
            vertices=[(left.time, left.price), (trough1.time, trough1.price), (head.time, head.price),
                      (trough2.time, trough2.price), (right.time, right.price)],
            trendlines=[neckline],
            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
        ))
    return occurrences


def find_inverse_head_and_shoulders(bars: pd.DataFrame, swings: List[SwingPoint]) -> List[ChartPatternOccurrence]:
    occurrences: List[ChartPatternOccurrence] = []
    for i in range(len(swings) - 4):
        window = swings[i:i + 5]
        if [s.kind for s in window] != ["low", "high", "low", "high", "low"]:
            continue
        left, peak1, head, peak2, right = window

        if not (head.price < left.price and head.price < right.price):
            continue
        pattern_height = max(peak1.price, peak2.price) - head.price
        if pattern_height <= 0:
            continue
        shoulder_diff = abs(left.price - right.price)
        if shoulder_diff > 0.5 * pattern_height:
            continue
        if (min(left.price, right.price) - head.price) < 0.1 * pattern_height:
            continue

        neckline = ((peak1.time, peak1.price), (peak2.time, peak2.price))
        breakout = _find_breakout_above_line(bars, right.time, *neckline)
        if breakout is None:
            continue

        symmetry_score = max(0.0, 1 - shoulder_diff / (0.5 * pattern_height))
        prominence = (min(left.price, right.price) - head.price) / pattern_height
        prominence_score = min(1.0, prominence / 0.3)
        confidence = round(100 * (0.5 * symmetry_score + 0.5 * prominence_score), 1)

        occurrences.append(ChartPatternOccurrence(
            name="inverse_head_and_shoulders", direction="bullish",
            vertices=[(left.time, left.price), (peak1.time, peak1.price), (head.time, head.price),
                      (peak2.time, peak2.price), (right.time, right.price)],
            trendlines=[neckline],
            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
        ))
    return occurrences


def find_double_top(bars: pd.DataFrame, swings: List[SwingPoint], tolerance: float = 0.15) -> List[ChartPatternOccurrence]:
    occurrences: List[ChartPatternOccurrence] = []
    for i in range(len(swings) - 2):
        peak1, trough, peak2 = swings[i:i + 3]
        if (peak1.kind, trough.kind, peak2.kind) != ("high", "low", "high"):
            continue
        height = peak1.price - trough.price
        if height <= 0:
            continue
        diff = abs(peak1.price - peak2.price)
        if diff > tolerance * height:
            continue

        breakout = _find_breakout_below_level(bars, peak2.time, trough.price)
        if breakout is None:
            continue

        confidence = round(100 * max(0.0, 1 - diff / (tolerance * height)), 1)
        occurrences.append(ChartPatternOccurrence(
            name="double_top", direction="bearish",
            vertices=[(peak1.time, peak1.price), (trough.time, trough.price), (peak2.time, peak2.price)],
            trendlines=[((trough.time, trough.price), (breakout[0], trough.price))],
            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
        ))
    return occurrences


def find_double_bottom(bars: pd.DataFrame, swings: List[SwingPoint], tolerance: float = 0.15) -> List[ChartPatternOccurrence]:
    occurrences: List[ChartPatternOccurrence] = []
    for i in range(len(swings) - 2):
        trough1, peak, trough2 = swings[i:i + 3]
        if (trough1.kind, peak.kind, trough2.kind) != ("low", "high", "low"):
            continue
        height = peak.price - trough1.price
        if height <= 0:
            continue
        diff = abs(trough1.price - trough2.price)
        if diff > tolerance * height:
            continue

        breakout = _find_breakout_above_level(bars, trough2.time, peak.price)
        if breakout is None:
            continue

        confidence = round(100 * max(0.0, 1 - diff / (tolerance * height)), 1)
        occurrences.append(ChartPatternOccurrence(
            name="double_bottom", direction="bullish",
            vertices=[(trough1.time, trough1.price), (peak.time, peak.price), (trough2.time, trough2.price)],
            trendlines=[((peak.time, peak.price), (breakout[0], peak.price))],
            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
        ))
    return occurrences


def _fit_trendline(bars: pd.DataFrame, points: List[SwingPoint]) -> Tuple[float, float]:
    """(slope, intercept) for price vs. bar position (not raw timestamp --
    keeps units interpretable as price-per-bar and avoids floating-point
    issues with nanosecond-scale timestamps)."""
    xs = np.array([bars.index.get_loc(p.time) for p in points], dtype=float)
    ys = np.array([p.price for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _scan_converging_patterns(
    bars: pd.DataFrame, swings: List[SwingPoint],
    lookback_bars: int = 60, step: int = 10, flat_threshold: float = 0.0005,
) -> List[ChartPatternOccurrence]:
    """Shared trendline-fitting scanner for triangles and wedges: within a
    rolling lookback window, fit a line through the recent swing highs and
    a separate line through the recent swing lows, classify by slope
    combination, and confirm on whichever line price actually breaks
    first (not a fixed assumed direction).

    flat_threshold is intentionally not razor-tight: two touches of the
    same resistance/support rarely land at the exact same price even in a
    textbook triangle, so a per-bar slope threshold that's too strict
    reclassifies a visually-flat line as a slight wedge slope instead."""
    occurrences: List[ChartPatternOccurrence] = []
    n = len(bars)
    avg_price = float(bars["close"].mean())
    if avg_price <= 0:
        return occurrences

    window_end = lookback_bars
    while window_end < n:
        window_start = window_end - lookback_bars
        start_time = bars.index[window_start]
        end_time = bars.index[window_end - 1]

        window_swings = [s for s in swings if start_time <= s.time <= end_time]
        highs = [s for s in window_swings if s.kind == "high"]
        lows = [s for s in window_swings if s.kind == "low"]

        if len(highs) >= 2 and len(lows) >= 2:
            high_slope, high_intercept = _fit_trendline(bars, highs)
            low_slope, low_intercept = _fit_trendline(bars, lows)
            norm_high = high_slope / avg_price
            norm_low = low_slope / avg_price
            high_flat = abs(norm_high) < flat_threshold

            pattern_name = None
            if high_flat and norm_low > flat_threshold:
                pattern_name = "ascending_triangle"
            elif abs(norm_low) < flat_threshold and norm_high < -flat_threshold:
                pattern_name = "descending_triangle"
            elif norm_high < -flat_threshold and norm_low > flat_threshold:
                pattern_name = "symmetrical_triangle"
            elif norm_high > flat_threshold and norm_low > flat_threshold and norm_high < norm_low:
                pattern_name = "rising_wedge"
            elif norm_high < -flat_threshold and norm_low < -flat_threshold and norm_high < norm_low:
                pattern_name = "falling_wedge"

            if pattern_name is not None:
                high_p1 = (bars.index[window_start], high_slope * window_start + high_intercept)
                high_p2 = (bars.index[window_end - 1], high_slope * (window_end - 1) + high_intercept)
                low_p1 = (bars.index[window_start], low_slope * window_start + low_intercept)
                low_p2 = (bars.index[window_end - 1], low_slope * (window_end - 1) + low_intercept)
                gap_start = high_p1[1] - low_p1[1]
                gap_end = high_p2[1] - low_p2[1]

                if gap_start > 0 and gap_end > 0 and gap_end < gap_start * 0.9:
                    breakout_up = _find_breakout_above_line(bars, end_time, high_p1, high_p2)
                    breakout_down = _find_breakout_below_line(bars, end_time, low_p1, low_p2)

                    breakout, direction = None, None
                    if breakout_up and breakout_down:
                        breakout, direction = (breakout_up, "bullish") if breakout_up[0] <= breakout_down[0] else (breakout_down, "bearish")
                    elif breakout_up:
                        breakout, direction = breakout_up, "bullish"
                    elif breakout_down:
                        breakout, direction = breakout_down, "bearish"

                    if breakout is not None:
                        confidence = round(100 * min(1.0, (gap_start - gap_end) / gap_start), 1)
                        occurrences.append(ChartPatternOccurrence(
                            name=pattern_name, direction=direction,
                            vertices=[(s.time, s.price) for s in window_swings],
                            trendlines=[(high_p1, high_p2), (low_p1, low_p2)],
                            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
                        ))
        window_end += step

    return occurrences


def find_flags(
    bars: pd.DataFrame, swings: List[SwingPoint],
    pole_lookback: int = 10, pole_min_pct: float = 5.0, channel_bars: int = 15,
) -> List[ChartPatternOccurrence]:
    """Sharp impulsive 'pole' move followed by a tight, roughly flat
    consolidation 'channel', confirmed on a breakout continuing the pole's
    direction. Doesn't use swing points -- included as a parameter for a
    uniform detector signature."""
    occurrences: List[ChartPatternOccurrence] = []
    closes = bars["close"]
    n = len(bars)
    i = pole_lookback

    while i < n - channel_bars - 1:
        pole_start_price = float(closes.iloc[i - pole_lookback])
        pole_end_price = float(closes.iloc[i])
        if pole_start_price <= 0:
            i += 1
            continue
        pole_pct = (pole_end_price / pole_start_price - 1) * 100

        if abs(pole_pct) < pole_min_pct:
            i += 1
            continue

        channel = bars.iloc[i:i + channel_bars]
        channel_high = float(channel["high"].max())
        channel_low = float(channel["low"].min())
        channel_mean = float(channel["close"].mean())
        channel_range_pct = (channel_high - channel_low) / channel_mean * 100 if channel_mean else 999.0

        if channel_range_pct > abs(pole_pct) * 0.5:
            i += 1
            continue

        direction = "bullish" if pole_pct > 0 else "bearish"
        pole_start_time = bars.index[i - pole_lookback]
        pole_end_time = bars.index[i]
        channel_end_time = channel.index[-1]

        if direction == "bullish":
            breakout = _find_breakout_above_level(bars, channel_end_time, channel_high)
            flag_vertex_price = channel_high
        else:
            breakout = _find_breakout_below_level(bars, channel_end_time, channel_low)
            flag_vertex_price = channel_low

        if breakout is None:
            i += 1
            continue

        tightness = max(0.0, 1 - channel_range_pct / (abs(pole_pct) * 0.5 + 1e-9))
        strength = min(1.0, abs(pole_pct) / 15.0)
        confidence = round(100 * (0.5 * tightness + 0.5 * strength), 1)

        occurrences.append(ChartPatternOccurrence(
            name="flag", direction=direction,
            vertices=[(pole_start_time, pole_start_price), (pole_end_time, pole_end_price),
                      (channel_end_time, flag_vertex_price)],
            trendlines=[((pole_end_time, channel_high), (channel_end_time, channel_high)),
                        ((pole_end_time, channel_low), (channel_end_time, channel_low))],
            breakout_time=breakout[0], breakout_price=breakout[1], confidence=confidence,
        ))
        i += channel_bars

    return occurrences


CHART_PATTERN_NAMES = [
    "head_and_shoulders", "inverse_head_and_shoulders", "double_top", "double_bottom",
    "ascending_triangle", "descending_triangle", "symmetrical_triangle",
    "rising_wedge", "falling_wedge", "flag",
]


# A single fixed window can only ever see patterns that fit entirely
# inside it, so a triangle/wedge spanning several months on daily bars
# (very common -- these patterns aren't confined to a few weeks) was
# structurally invisible when only lookback_bars=60 was ever scanned.
# Scanning several scales lets short, medium, and long-forming patterns
# all surface instead of only the shortest.
CONVERGING_PATTERN_SCALES: Tuple[int, ...] = (60, 120, 250)

# Same disease as the triangle scanner: a single fixed (pole_lookback,
# channel_bars) pair only ever finds flags of essentially one duration --
# empirically, every occurrence landed within a few days of one narrow
# span regardless of symbol. Small/medium/large pairs (channel roughly
# 1-1.5x the pole, a common flag convention) surface genuinely varied
# pole/consolidation sizes instead.
FLAG_PATTERN_SCALES: Tuple[Tuple[int, int], ...] = ((5, 8), (10, 15), (20, 25))


def find_all_chart_patterns(bars: pd.DataFrame) -> Dict[str, List[ChartPatternOccurrence]]:
    """Single entry point used by everything downstream (strategy, stats,
    dashboard payload, CLI status) -- computes swing points and the shared
    triangle/wedge scan exactly once rather than redundantly per pattern
    type."""
    swings = find_swing_points(bars)
    flags: List[ChartPatternOccurrence] = []
    for pole_lookback, channel_bars in FLAG_PATTERN_SCALES:
        flags.extend(find_flags(bars, swings, pole_lookback=pole_lookback, channel_bars=channel_bars))
    result: Dict[str, List[ChartPatternOccurrence]] = {
        "head_and_shoulders": find_head_and_shoulders(bars, swings),
        "inverse_head_and_shoulders": find_inverse_head_and_shoulders(bars, swings),
        "double_top": find_double_top(bars, swings),
        "double_bottom": find_double_bottom(bars, swings),
        "flag": flags,
    }
    converging: List[ChartPatternOccurrence] = []
    for lookback_bars in CONVERGING_PATTERN_SCALES:
        converging.extend(_scan_converging_patterns(bars, swings, lookback_bars=lookback_bars))
    for name in ("ascending_triangle", "descending_triangle", "symmetrical_triangle", "rising_wedge", "falling_wedge"):
        result[name] = [o for o in converging if o.name == name]
    return result
