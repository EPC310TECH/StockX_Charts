from dataclasses import dataclass
from typing import List

import pandas as pd

from stockx.analysis.patterns import (
    MIN_OCCURRENCES_FOR_WIN_RATE,
    compute_chart_pattern_stats,
    compute_pattern_stats,
    current_pattern_state,
)
from stockx.data.cache import get_bars
from stockx.exceptions import DataFetchError, InsufficientHistoryError
from stockx.strategies.chart_patterns import find_all_chart_patterns

DEFAULT_MIN_WIN_RATE = 0.55


@dataclass
class ScanHit:
    symbol: str
    pattern_name: str
    pattern_kind: str  # "candlestick" | "chart"
    direction: str
    win_rate: float
    occurrences: int
    avg_confidence: float
    signal_time: pd.Timestamp


def _scan_symbol(symbol: str, interval: str, min_win_rate: float, min_occurrences: int) -> List[ScanHit]:
    """A hit is 'this pattern just fired on this symbol, and historically
    -- on this symbol specifically, not textbook averages -- it's won often
    enough, on enough samples, to trust.' Reuses whatever's already cached
    (refresh=False) so scanning a whole watchlist doesn't fire one live
    fetch per symbol."""
    bars = get_bars(symbol, interval=interval, refresh=False)
    state = current_pattern_state(bars, symbol)

    hits: List[ScanHit] = []

    pattern_stats = {s.name: s for s in compute_pattern_stats(bars)}
    for occ in state.recent_confirmed:
        stats = pattern_stats.get(occ.name)
        if not stats or stats.occurrences < min_occurrences or pd.isna(stats.win_rate):
            continue
        if stats.win_rate >= min_win_rate:
            hits.append(ScanHit(
                symbol=symbol, pattern_name=occ.name, pattern_kind="candlestick",
                direction=stats.direction, win_rate=stats.win_rate,
                occurrences=stats.occurrences, avg_confidence=stats.avg_confidence,
                signal_time=occ.timestamp,
            ))

    all_chart_patterns = find_all_chart_patterns(bars)
    chart_pattern_stats = {s.name: s for s in compute_chart_pattern_stats(bars, all_chart_patterns)}
    for occ in state.recent_chart_pattern_breakouts:
        stats = chart_pattern_stats.get(occ.name)
        if not stats or stats.occurrences < min_occurrences or pd.isna(stats.win_rate):
            continue
        if stats.win_rate >= min_win_rate:
            hits.append(ScanHit(
                symbol=symbol, pattern_name=occ.name, pattern_kind="chart",
                direction=occ.direction, win_rate=stats.win_rate,
                occurrences=stats.occurrences, avg_confidence=stats.avg_confidence,
                signal_time=occ.timestamp,
            ))

    return hits


def scan_watchlist(
    symbols: List[str],
    interval: str,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_occurrences: int = MIN_OCCURRENCES_FOR_WIN_RATE,
) -> dict:
    """Runs every watchlist symbol through _scan_symbol, skipping (not
    failing) symbols with no cached data yet at this interval -- a scan
    should return honest partial results, not 500 because one watched
    symbol was never opened at this timeframe."""
    hits: List[ScanHit] = []
    skipped: List[str] = []

    for symbol in symbols:
        try:
            hits.extend(_scan_symbol(symbol, interval, min_win_rate, min_occurrences))
        except (DataFetchError, InsufficientHistoryError):
            skipped.append(symbol)

    hits.sort(key=lambda h: -h.win_rate)

    return {
        "interval": interval,
        "min_win_rate": min_win_rate,
        "scanned": len(symbols) - len(skipped),
        "skipped": skipped,
        "hits": [
            {
                "symbol": h.symbol,
                "pattern_name": h.pattern_name,
                "pattern_kind": h.pattern_kind,
                "direction": h.direction,
                "win_rate": h.win_rate,
                "occurrences": h.occurrences,
                "avg_confidence": h.avg_confidence,
                "signal_time": int(h.signal_time.timestamp()),
            }
            for h in hits
        ],
    }
