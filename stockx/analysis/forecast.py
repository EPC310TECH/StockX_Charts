from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from stockx.analysis.patterns import MIN_OCCURRENCES_FOR_WIN_RATE, PATTERN_REGISTRY, PatternStats
from stockx.data.cache import get_bars
from stockx.strategies import indicators as ind

# Prototype: a deterministic, fully-explainable "next N bars" directional
# forecast built entirely from signals this project already computes and
# already trusts -- candlestick patterns (PATTERN_REGISTRY) plus a small
# set of indicator-derived boolean conditions below, each weighted by its
# OWN empirically-measured win rate rather than a hand-picked number. This
# intentionally mirrors compute_pattern_stats' methodology (same
# forward-return-direction-match definition of "win") so pattern and
# indicator signals are scored on the same footing.
#
# Deliberately NOT included yet, to keep this prototype small and fast to
# iterate on: chart patterns (find_all_chart_patterns is comparatively
# expensive to run per-fold) and any ML/interaction modeling. See
# walk_forward_validate_forecast's docstring for why validation -- not the
# scoring formula -- is the actual point of this module.

INDICATOR_SIGNAL_REGISTRY: List[Tuple[str, callable, str]] = [
    ("rsi_oversold", lambda bars: ind.rsi(bars["close"]) < 30, "bullish"),
    ("rsi_overbought", lambda bars: ind.rsi(bars["close"]) > 70, "bearish"),
    ("macd_bullish", lambda bars: ind.macd(bars["close"])[0] > ind.macd(bars["close"])[1], "bullish"),
    ("macd_bearish", lambda bars: ind.macd(bars["close"])[0] < ind.macd(bars["close"])[1], "bearish"),
    ("bollinger_upper_break", lambda bars: bars["close"] > ind.bollinger_bands(bars["close"])[0], "bullish"),
    ("bollinger_lower_break", lambda bars: bars["close"] < ind.bollinger_bands(bars["close"])[2], "bearish"),
    ("stochastic_oversold", lambda bars: ind.stochastic(bars)[0] < 20, "bullish"),
    ("stochastic_overbought", lambda bars: ind.stochastic(bars)[0] > 80, "bearish"),
    ("awesome_oscillator_positive", lambda bars: ind.awesome_oscillator(bars) > 0, "bullish"),
    ("awesome_oscillator_negative", lambda bars: ind.awesome_oscillator(bars) < 0, "bearish"),
    ("above_parabolic_sar", lambda bars: bars["close"] > ind.parabolic_sar(bars), "bullish"),
    ("below_parabolic_sar", lambda bars: bars["close"] < ind.parabolic_sar(bars), "bearish"),
    # Volume-derived: none of the above use volume at all, only price
    # shape/level. These read conviction (or lack of it) behind a move.
    ("mfi_oversold", lambda bars: ind.money_flow_index(bars) < 20, "bullish"),
    ("mfi_overbought", lambda bars: ind.money_flow_index(bars) > 80, "bearish"),
    ("obv_rising", lambda bars: ind.on_balance_volume(bars) > ind.on_balance_volume(bars).rolling(20).mean(), "bullish"),
    ("obv_falling", lambda bars: ind.on_balance_volume(bars) < ind.on_balance_volume(bars).rolling(20).mean(), "bearish"),
    ("chaikin_money_flow_positive", lambda bars: ind.chaikin_money_flow(bars) > 0, "bullish"),
    ("chaikin_money_flow_negative", lambda bars: ind.chaikin_money_flow(bars) < 0, "bearish"),
    ("volume_surge_up", lambda bars: ind.volume_surge(bars) & (bars["close"] > bars["open"]), "bullish"),
    ("volume_surge_down", lambda bars: ind.volume_surge(bars) & (bars["close"] < bars["open"]), "bearish"),
]


def compute_indicator_signal_stats(bars: pd.DataFrame, forward_bars: int = 10) -> List[PatternStats]:
    """Empirical reliability per indicator signal, structurally identical
    to patterns.compute_pattern_stats: of this condition's past bars on
    this symbol, what fraction were followed by a move in the condition's
    stated direction over the next `forward_bars` bars."""
    forward_return = bars["close"].shift(-forward_bars) / bars["close"] - 1
    has_forward_data = forward_return.notna()

    stats: List[PatternStats] = []
    for name, bool_fn, direction in INDICATOR_SIGNAL_REGISTRY:
        occurred = bool_fn(bars).fillna(False) & has_forward_data
        n = int(occurred.sum())
        if n == 0:
            stats.append(PatternStats(name, direction, 0, float("nan"), float("nan"), float("nan")))
            continue
        returns = forward_return[occurred]
        win_rate = float((returns > 0).mean()) if direction == "bullish" else float((returns < 0).mean())
        stats.append(PatternStats(
            name=name, direction=direction, occurrences=n,
            win_rate=win_rate, avg_forward_return=float(returns.mean()), avg_confidence=float("nan"),
        ))
    return stats


@dataclass
class CompositeForecast:
    as_of: pd.Timestamp
    score: float  # -1 (max bearish) .. 0 (no opinion) .. +1 (max bullish)
    contributing: List[Tuple[str, str, float, int]]  # (name, direction, weight, occurrences)


def _signal_weight(stat: PatternStats) -> float:
    """0.5 win rate (no edge over a coin flip) -> 0 weight; 1.0 win rate
    -> full weight in the signal's own stated direction. Signals with too
    few historical occurrences to trust (below MIN_OCCURRENCES_FOR_WIN_RATE,
    the same bar used elsewhere in this project) are excluded entirely
    rather than given a small noisy weight."""
    if stat.direction not in ("bullish", "bearish"):
        return 0.0  # e.g. doji's "neutral" -- no directional claim to weight
    if pd.isna(stat.win_rate) or stat.occurrences < MIN_OCCURRENCES_FOR_WIN_RATE:
        return 0.0
    sign = 1.0 if stat.direction == "bullish" else -1.0
    return sign * max(0.0, (stat.win_rate - 0.5) * 2)


def compute_composite_score(
    bars: pd.DataFrame,
    pattern_stats: List[PatternStats],
    indicator_stats: List[PatternStats],
) -> CompositeForecast:
    """The forecast for the single most recent bar in `bars`: every
    pattern/indicator signal that's true on that bar, weighted by its
    (caller-supplied, presumably train-window-only) win rate, averaged.
    `bars` must have enough history before its last row for the
    indicators' own warmup (e.g. rolling/EWM windows) to be valid -- the
    same requirement any of these functions already have."""
    stats_by_name = {s.name: s for s in pattern_stats + indicator_stats}
    contributing: List[Tuple[str, str, float, int]] = []

    for name, bool_fn, _confidence_fn, direction in PATTERN_REGISTRY:
        stat = stats_by_name.get(name)
        if stat is None or direction not in ("bullish", "bearish"):
            continue
        if bool(bool_fn(bars).iloc[-1]):
            weight = _signal_weight(stat)
            if weight != 0.0:
                contributing.append((name, direction, weight, stat.occurrences))

    for name, bool_fn, _direction in INDICATOR_SIGNAL_REGISTRY:
        stat = stats_by_name.get(name)
        if stat is None:
            continue
        if bool(bool_fn(bars).fillna(False).iloc[-1]):
            weight = _signal_weight(stat)
            if weight != 0.0:
                contributing.append((name, stat.direction, weight, stat.occurrences))

    score = float(np.mean([w for _, _, w, _ in contributing])) if contributing else 0.0
    return CompositeForecast(as_of=bars.index[-1], score=score, contributing=contributing)


@dataclass
class ForecastFold:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_test_bars: int
    n_predictions: int
    coverage: float
    hit_rate: float
    baseline_up_rate: float


@dataclass
class ForecastValidationReport:
    symbol: str
    interval: str
    forward_bars: int
    folds: List[ForecastFold]
    overall_hit_rate: float
    overall_coverage: float
    overall_baseline_up_rate: float
    total_predictions: int


def walk_forward_validate_forecast(
    symbol: str,
    interval: str = "1h",
    forward_bars: int = 1,
    train_days: int = 60,
    test_days: int = 20,
    refresh: bool = True,
) -> ForecastValidationReport:
    """The actual point of this module: pattern/indicator win rates
    (PatternStats) are measured *in-sample* by construction -- computed
    from the very history being described. Using them as forward-looking
    forecast weights without checking is exactly the kind of lookahead-
    flavored overconfidence this project has caught and fixed elsewhere
    this session. This function is the check: weights are measured on a
    train window only, then applied to the immediately-following,
    never-seen test window, walked forward through the whole history in
    non-overlapping folds -- same fold structure as run_walk_forward.

    Every pattern/indicator boolean function is already causal (no
    lookahead -- verified throughout this project), so it's correct (not
    just faster) to compute each signal's boolean Series once, vectorized,
    over the full bars series and then index into it per-bar, rather than
    re-slicing and recomputing bars for every single test bar.
    """
    min_days = train_days + test_days * 2
    bars = get_bars(symbol, interval=interval, refresh=refresh, min_days=min_days)
    trading_days = sorted(pd.unique(bars.index.date))

    forward_return = bars["close"].shift(-forward_bars) / bars["close"] - 1

    # Precompute every signal's boolean Series once over the whole history
    # (see docstring) -- reused across every fold's test evaluation.
    pattern_signals = {name: bool_fn(bars).fillna(False) for name, bool_fn, _cf, _d in PATTERN_REGISTRY}
    indicator_signals = {name: bool_fn(bars).fillna(False) for name, bool_fn, _d in INDICATOR_SIGNAL_REGISTRY}

    folds: List[ForecastFold] = []
    all_hits: List[bool] = []
    all_up_actuals: List[bool] = []
    i = 0
    fold_id = 0
    while i + train_days + test_days <= len(trading_days):
        train_day_slice = trading_days[i:i + train_days]
        test_day_slice = trading_days[i + train_days:i + train_days + test_days]

        train_mask = (bars.index.date >= train_day_slice[0]) & (bars.index.date <= train_day_slice[-1])
        test_mask = (bars.index.date >= test_day_slice[0]) & (bars.index.date <= test_day_slice[-1])
        train_bars = bars[train_mask]
        test_bars = bars[test_mask]

        if train_bars.empty or test_bars.empty:
            i += test_days
            continue

        pattern_stats = _stats_from_precomputed(pattern_signals, forward_return, train_bars.index, PATTERN_REGISTRY)
        indicator_stats = _stats_from_precomputed(
            indicator_signals, forward_return, train_bars.index, INDICATOR_SIGNAL_REGISTRY
        )
        weights = {s.name: _signal_weight(s) for s in pattern_stats + indicator_stats}

        fold_hits: List[bool] = []
        fold_up_actuals: List[bool] = []
        for ts in test_bars.index:
            fr = forward_return.loc[ts]
            if pd.isna(fr):
                continue
            active_weights = [
                weights.get(name, 0.0)
                for name, series in {**pattern_signals, **indicator_signals}.items()
                if series.loc[ts] and weights.get(name, 0.0) != 0.0
            ]
            if not active_weights:
                continue
            score = float(np.mean(active_weights))
            predicted_up = score > 0
            actual_up = fr > 0
            fold_hits.append(predicted_up == actual_up)
            fold_up_actuals.append(actual_up)

        n_predictions = len(fold_hits)
        folds.append(ForecastFold(
            fold_id=fold_id,
            train_start=train_day_slice[0], train_end=train_day_slice[-1],
            test_start=test_day_slice[0], test_end=test_day_slice[-1],
            n_test_bars=len(test_bars),
            n_predictions=n_predictions,
            coverage=(n_predictions / len(test_bars)) if len(test_bars) else 0.0,
            hit_rate=(sum(fold_hits) / n_predictions) if n_predictions else float("nan"),
            baseline_up_rate=(sum(fold_up_actuals) / n_predictions) if n_predictions else float("nan"),
        ))
        all_hits.extend(fold_hits)
        all_up_actuals.extend(fold_up_actuals)
        fold_id += 1
        i += test_days

    total_predictions = len(all_hits)
    total_test_bars = sum(f.n_test_bars for f in folds)
    return ForecastValidationReport(
        symbol=symbol.upper(),
        interval=interval,
        forward_bars=forward_bars,
        folds=folds,
        overall_hit_rate=(sum(all_hits) / total_predictions) if total_predictions else float("nan"),
        overall_coverage=(total_predictions / total_test_bars) if total_test_bars else 0.0,
        overall_baseline_up_rate=(sum(all_up_actuals) / total_predictions) if total_predictions else float("nan"),
        total_predictions=total_predictions,
    )


def _stats_from_precomputed(signals: dict, forward_return: pd.Series, train_index, registry) -> List[PatternStats]:
    """Same win-rate methodology as compute_pattern_stats/
    compute_indicator_signal_stats, but sliced from the full-history
    precomputed signal Series down to a train window's index, instead of
    recomputing the signal function from scratch on a train-only slice --
    equivalent by causality, much cheaper across many folds."""
    stats: List[PatternStats] = []
    for entry in registry:
        name, direction = entry[0], entry[-1]
        occurred_full = signals[name]
        occurred = occurred_full.loc[train_index] & forward_return.loc[train_index].notna()
        n = int(occurred.sum())
        if n == 0:
            stats.append(PatternStats(name, direction, 0, float("nan"), float("nan"), float("nan")))
            continue
        returns = forward_return.loc[train_index][occurred]
        if direction == "bullish":
            win_rate = float((returns > 0).mean())
        elif direction == "bearish":
            win_rate = float((returns < 0).mean())
        else:
            win_rate = float("nan")
        stats.append(PatternStats(
            name=name, direction=direction, occurrences=n,
            win_rate=win_rate, avg_forward_return=float(returns.mean()), avg_confidence=float("nan"),
        ))
    return stats
