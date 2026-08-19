from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from stockx.strategies import candlestick_patterns as cp
from stockx.strategies.chart_patterns import ChartPatternOccurrence, find_all_chart_patterns

# Single source of truth: (name, boolean detector, confidence score, direction).
# Used by current_pattern_state (live status) and compute_pattern_stats
# (historical reliability) so the pattern list is never duplicated.
PATTERN_REGISTRY: List[Tuple[str, callable, callable, str]] = [
    ("doji", cp.is_doji, cp.doji_confidence, "neutral"),
    ("hammer", cp.is_hammer, cp.hammer_confidence, "bullish"),
    ("hanging_man", cp.is_hanging_man, cp.hanging_man_confidence, "bearish"),
    ("shooting_star", cp.is_shooting_star, cp.shooting_star_confidence, "bearish"),
    ("inverted_hammer", cp.is_inverted_hammer, cp.inverted_hammer_confidence, "bullish"),
    ("bullish_engulfing", cp.is_bullish_engulfing, cp.bullish_engulfing_confidence, "bullish"),
    ("bearish_engulfing", cp.is_bearish_engulfing, cp.bearish_engulfing_confidence, "bearish"),
    ("morning_star", cp.is_morning_star, cp.morning_star_confidence, "bullish"),
    ("evening_star", cp.is_evening_star, cp.evening_star_confidence, "bearish"),
    ("three_white_soldiers", cp.is_three_white_soldiers, cp.three_white_soldiers_confidence, "bullish"),
    ("three_black_crows", cp.is_three_black_crows, cp.three_black_crows_confidence, "bearish"),
]

FORMING_PATTERNS = [
    ("morning_star", cp.morning_star_forming),
    ("evening_star", cp.evening_star_forming),
]

MIN_OCCURRENCES_FOR_WIN_RATE = 5


@dataclass
class PatternOccurrence:
    name: str
    timestamp: pd.Timestamp


@dataclass
class ChartPatternBreakoutOccurrence:
    name: str
    direction: str
    timestamp: pd.Timestamp
    confidence: float


@dataclass
class PatternState:
    symbol: str
    as_of: pd.Timestamp
    recent_confirmed: List[PatternOccurrence]
    forming: List[str]
    in_contraction: bool
    recent_chart_pattern_breakouts: List[ChartPatternBreakoutOccurrence]


@dataclass
class PatternStats:
    name: str
    direction: str
    occurrences: int
    win_rate: float
    avg_forward_return: float
    avg_confidence: float


def compute_pattern_stats(bars: pd.DataFrame, forward_bars: int = 10) -> List[PatternStats]:
    """Empirical reliability per pattern type: of this pattern's past
    occurrences on this symbol, what fraction were followed by a move in
    the pattern's stated direction over the next `forward_bars` bars.
    Only counts occurrences where `forward_bars` of future data actually
    exists (no fabricated stats off the tail of the series)."""
    forward_return = bars["close"].shift(-forward_bars) / bars["close"] - 1
    has_forward_data = forward_return.notna()

    stats: List[PatternStats] = []
    for name, bool_fn, confidence_fn, direction in PATTERN_REGISTRY:
        occurred = bool_fn(bars) & has_forward_data
        n = int(occurred.sum())
        if n == 0:
            stats.append(PatternStats(name, direction, 0, float("nan"), float("nan"), float("nan")))
            continue

        returns = forward_return[occurred]
        if direction == "bullish":
            win_rate = float((returns > 0).mean())
        elif direction == "bearish":
            win_rate = float((returns < 0).mean())
        else:
            win_rate = float("nan")

        stats.append(PatternStats(
            name=name,
            direction=direction,
            occurrences=n,
            win_rate=win_rate,
            avg_forward_return=float(returns.mean()),
            avg_confidence=float(confidence_fn(bars)[occurred].mean()),
        ))
    return stats


def compute_chart_pattern_stats(
    bars: pd.DataFrame,
    all_chart_patterns: Dict[str, List[ChartPatternOccurrence]] = None,
    forward_bars: int = 10,
) -> List[PatternStats]:
    """Empirical reliability per chart-pattern type, structurally the same
    as compute_pattern_stats -- just anchored at each occurrence's
    breakout_time instead of a boolean-flagged bar, and win = the forward
    return actually moving in that occurrence's own breakout direction
    (triangles/wedges can break either way, so this is checked per
    occurrence, not by a fixed pattern-level direction)."""
    if all_chart_patterns is None:
        all_chart_patterns = find_all_chart_patterns(bars)

    forward_return = bars["close"].shift(-forward_bars) / bars["close"] - 1

    stats: List[PatternStats] = []
    for name, occurrences in all_chart_patterns.items():
        directions = [occ.direction for occ in occurrences]
        primary_direction = max(set(directions), key=directions.count) if directions else "neutral"

        returns, confidences, wins, n = [], [], 0, 0
        for occ in occurrences:
            if occ.breakout_time not in forward_return.index:
                continue
            fr = forward_return.loc[occ.breakout_time]
            if pd.isna(fr):
                continue
            n += 1
            returns.append(fr)
            confidences.append(occ.confidence)
            if (occ.direction == "bullish" and fr > 0) or (occ.direction == "bearish" and fr < 0):
                wins += 1

        if n == 0:
            stats.append(PatternStats(name, primary_direction, 0, float("nan"), float("nan"), float("nan")))
            continue

        stats.append(PatternStats(
            name=name, direction=primary_direction, occurrences=n,
            win_rate=wins / n, avg_forward_return=float(sum(returns) / n),
            avg_confidence=float(sum(confidences) / n),
        ))
    return stats


def current_pattern_state(bars: pd.DataFrame, symbol: str, lookback_bars: int = 5) -> PatternState:
    recent_confirmed: List[PatternOccurrence] = []
    for name, fn, _confidence_fn, _direction in PATTERN_REGISTRY:
        flags = fn(bars).iloc[-lookback_bars:]
        for ts, flag in flags.items():
            if bool(flag):
                recent_confirmed.append(PatternOccurrence(name=name, timestamp=ts))
    recent_confirmed.sort(key=lambda occ: occ.timestamp)

    forming = [name for name, fn in FORMING_PATTERNS if bool(fn(bars).iloc[-1])]
    in_contraction = bool(cp.range_contraction(bars).iloc[-1])

    cutoff_time = bars.index[-lookback_bars] if len(bars) >= lookback_bars else bars.index[0]
    all_chart_patterns = find_all_chart_patterns(bars)
    recent_chart_breakouts: List[ChartPatternBreakoutOccurrence] = []
    for name, occurrences in all_chart_patterns.items():
        for occ in occurrences:
            if occ.breakout_time >= cutoff_time:
                recent_chart_breakouts.append(
                    ChartPatternBreakoutOccurrence(name, occ.direction, occ.breakout_time, occ.confidence)
                )
    recent_chart_breakouts.sort(key=lambda occ: occ.timestamp)

    return PatternState(
        symbol=symbol.upper(),
        as_of=bars.index[-1],
        recent_confirmed=recent_confirmed,
        forming=forming,
        in_contraction=in_contraction,
        recent_chart_pattern_breakouts=recent_chart_breakouts,
    )


def print_pattern_state(state: PatternState) -> None:
    print(f"\nCandlestick pattern status for {state.symbol}")
    print(f"  as of: {state.as_of}")

    if state.recent_confirmed:
        print("\n  Recently confirmed patterns:")
        for occ in state.recent_confirmed:
            print(f"    {occ.timestamp}  {occ.name}")
    else:
        print("\n  No confirmed patterns in the recent lookback window.")

    if state.forming:
        print("\n  Patterns forming (awaiting confirmation candle):")
        for name in state.forming:
            print(f"    {name}")
    else:
        print("\n  No multi-candle patterns currently forming.")

    contraction_note = (
        "YES -- range compressed vs its own recent history (breakout watch)"
        if state.in_contraction else "no"
    )
    print(f"\n  Range contraction (coiling): {contraction_note}")

    if state.recent_chart_pattern_breakouts:
        print("\n  Recent chart pattern breakouts:")
        for occ in state.recent_chart_pattern_breakouts:
            print(f"    {occ.timestamp}  {occ.name} ({occ.direction}, confidence {occ.confidence:.0f}/100)")
    else:
        print("\n  No chart pattern breakouts in the recent lookback window.")
