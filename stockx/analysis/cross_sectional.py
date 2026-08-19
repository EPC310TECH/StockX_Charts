from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stockx.analysis.forecast import INDICATOR_SIGNAL_REGISTRY, _signal_weight
from stockx.analysis.patterns import PATTERN_REGISTRY, PatternStats
from stockx.data.cache import get_bars

# The cross-sectional reframing of forecast.py's single-symbol prototype,
# following how professional quant shops actually approach this (see
# conversation): don't ask "will this one stock go up" (dominated by
# market-wide noise) -- ask "which stocks in my universe does the
# composite score currently favor, and do the favored ones actually
# outperform the disfavored ones going forward." Signal win rates are
# POOLED across every symbol in the universe during the train window
# (not measured per-symbol) -- one shared set of weights applied
# identically to every symbol, both because that's the standard
# cross-sectional approach (a signal should mean the same thing across
# names) and because pooling multiplies the occurrence count used to
# estimate each signal's reliability, directly addressing the
# too-few-folds problem the single-symbol version ran into.


@dataclass
class CrossSectionalFold:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_timestamps: int
    mean_ic: float
    ic_hit_rate: float  # fraction of test timestamps where the rank correlation was positive
    top_minus_bottom_spread: float  # avg (top-half fwd return - bottom-half fwd return) across test timestamps


@dataclass
class CrossSectionalReport:
    symbols: List[str]
    interval: str
    forward_bars: int
    folds: List[CrossSectionalFold]
    overall_mean_ic: float
    overall_ic_hit_rate: float
    overall_top_minus_bottom_spread: float
    total_timestamps: int


def _pooled_stats(
    signals_by_symbol: Dict[str, Dict[str, pd.Series]],
    forward_return_by_symbol: Dict[str, pd.Series],
    train_days_by_symbol: Dict[str, np.ndarray],
    registry,
) -> List[PatternStats]:
    """Same win-rate methodology as forecast.py's stats functions, but
    pooling occurrences and forward returns across every symbol's train
    window into one combined sample per signal, instead of one sample per
    symbol."""
    stats: List[PatternStats] = []
    for entry in registry:
        name, direction = entry[0], entry[-1]
        pooled_returns = []
        for symbol, signals in signals_by_symbol.items():
            train_days = train_days_by_symbol[symbol]
            fr = forward_return_by_symbol[symbol]
            occurred_full = signals[name]
            train_mask = pd.Series(fr.index.date, index=fr.index).isin(train_days)
            occurred = occurred_full & train_mask & fr.notna()
            if occurred.any():
                pooled_returns.append(fr[occurred])

        if not pooled_returns:
            stats.append(PatternStats(name, direction, 0, float("nan"), float("nan"), float("nan")))
            continue
        returns = pd.concat(pooled_returns)
        n = len(returns)
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


def walk_forward_validate_cross_sectional(
    symbols: List[str],
    interval: str = "1h",
    forward_bars: int = 1,
    train_days: int = 60,
    test_days: int = 20,
    min_symbols_per_timestamp: int = 5,
    refresh: bool = False,
) -> CrossSectionalReport:
    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            bars_by_symbol[symbol] = get_bars(symbol, interval=interval, refresh=refresh, min_days=train_days + test_days * 2)
        except Exception:
            continue  # a thin/unavailable symbol just drops out of the universe, not a hard failure

    if len(bars_by_symbol) < min_symbols_per_timestamp:
        raise ValueError(
            f"only {len(bars_by_symbol)} of {len(symbols)} symbols had enough history -- "
            f"need at least {min_symbols_per_timestamp} for a meaningful cross-section"
        )

    # Fold boundaries walk forward over the days common to every symbol in
    # the universe, so every fold's train/test window means the same
    # calendar period for all of them.
    common_days = None
    for bars in bars_by_symbol.values():
        days = set(bars.index.date)
        common_days = days if common_days is None else (common_days & days)
    trading_days = sorted(common_days)

    forward_return_by_symbol = {s: b["close"].shift(-forward_bars) / b["close"] - 1 for s, b in bars_by_symbol.items()}
    pattern_signals_by_symbol = {
        s: {name: bool_fn(b).fillna(False) for name, bool_fn, _cf, _d in PATTERN_REGISTRY}
        for s, b in bars_by_symbol.items()
    }
    indicator_signals_by_symbol = {
        s: {name: bool_fn(b).fillna(False) for name, bool_fn, _d in INDICATOR_SIGNAL_REGISTRY}
        for s, b in bars_by_symbol.items()
    }
    all_signals_by_symbol = {
        s: {**pattern_signals_by_symbol[s], **indicator_signals_by_symbol[s]} for s in bars_by_symbol
    }

    folds: List[CrossSectionalFold] = []
    all_ics: List[float] = []
    all_spreads: List[float] = []
    i = 0
    fold_id = 0
    while i + train_days + test_days <= len(trading_days):
        train_day_slice = np.array(trading_days[i:i + train_days])
        test_day_slice = trading_days[i + train_days:i + train_days + test_days]
        train_days_by_symbol = {s: train_day_slice for s in bars_by_symbol}

        pattern_stats = _pooled_stats(pattern_signals_by_symbol, forward_return_by_symbol, train_days_by_symbol, PATTERN_REGISTRY)
        indicator_stats = _pooled_stats(indicator_signals_by_symbol, forward_return_by_symbol, train_days_by_symbol, INDICATOR_SIGNAL_REGISTRY)
        weights = {s.name: _signal_weight(s) for s in pattern_stats + indicator_stats}

        # Common bar timestamps across every symbol's test window -- so
        # each cross-section compares symbols at the exact same moment.
        test_mask_by_symbol = {
            s: (b.index.date >= test_day_slice[0]) & (b.index.date <= test_day_slice[-1])
            for s, b in bars_by_symbol.items()
        }
        common_ts = None
        for s, mask in test_mask_by_symbol.items():
            ts = set(bars_by_symbol[s].index[mask])
            common_ts = ts if common_ts is None else (common_ts & ts)
        test_timestamps = sorted(common_ts) if common_ts else []

        fold_ics: List[float] = []
        fold_spreads: List[float] = []
        for ts in test_timestamps:
            scores, returns = [], []
            for symbol, signals in all_signals_by_symbol.items():
                fr = forward_return_by_symbol[symbol].get(ts)
                if fr is None or pd.isna(fr):
                    continue
                active_weights = [weights[name] for name, series in signals.items() if series.get(ts, False) and weights.get(name, 0.0) != 0.0]
                score = float(np.mean(active_weights)) if active_weights else 0.0
                scores.append(score)
                returns.append(fr)

            if len(scores) < min_symbols_per_timestamp:
                continue
            score_series = pd.Series(scores)
            return_series = pd.Series(returns)
            if score_series.nunique() < 2:
                continue  # every symbol scored identically (e.g. all zero) -- no ranking to evaluate

            # Spearman rank correlation without a scipy dependency: it's
            # exactly Pearson correlation computed on the rank-transformed
            # values.
            ic = score_series.rank().corr(return_series.rank())
            if pd.isna(ic):
                continue
            fold_ics.append(ic)

            median_score = score_series.median()
            top_returns = return_series[score_series > median_score]
            bottom_returns = return_series[score_series < median_score]
            if len(top_returns) and len(bottom_returns):
                fold_spreads.append(float(top_returns.mean() - bottom_returns.mean()))

        n_timestamps = len(fold_ics)
        folds.append(CrossSectionalFold(
            fold_id=fold_id,
            train_start=train_day_slice[0], train_end=train_day_slice[-1],
            test_start=test_day_slice[0], test_end=test_day_slice[-1],
            n_timestamps=n_timestamps,
            mean_ic=float(np.mean(fold_ics)) if fold_ics else float("nan"),
            ic_hit_rate=float(np.mean([ic > 0 for ic in fold_ics])) if fold_ics else float("nan"),
            top_minus_bottom_spread=float(np.mean(fold_spreads)) if fold_spreads else float("nan"),
        ))
        all_ics.extend(fold_ics)
        all_spreads.extend(fold_spreads)
        fold_id += 1
        i += test_days

    return CrossSectionalReport(
        symbols=sorted(bars_by_symbol.keys()),
        interval=interval,
        forward_bars=forward_bars,
        folds=folds,
        overall_mean_ic=float(np.mean(all_ics)) if all_ics else float("nan"),
        overall_ic_hit_rate=float(np.mean([ic > 0 for ic in all_ics])) if all_ics else float("nan"),
        overall_top_minus_bottom_spread=float(np.mean(all_spreads)) if all_spreads else float("nan"),
        total_timestamps=len(all_ics),
    )
