from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from stockx.analysis.regime import classify_regime
from stockx.backtest.engine import BacktestEngine, EngineConfig
from stockx.data.cache import get_bars
from stockx.metrics.metrics import Metrics, compute_metrics
from stockx.strategies import get_registered_strategies
from stockx.strategies.base import Strategy


@dataclass
class WalkForwardFold:
    fold_id: int
    regime_start: date
    regime_end: date
    test_start: date
    test_end: date
    trend_regime: str
    vol_regime: str
    strategy_metrics: Dict[str, Metrics]
    winner: Optional[str]


@dataclass
class WalkForwardReport:
    symbol: str
    interval: str
    rank_metric: str
    folds: List[WalkForwardFold]
    regime_strategy_map: Dict[Tuple[str, str], Dict[str, dict]]
    current_regime: Tuple[str, str]
    current_recommendation: Optional[str]
    bars: pd.DataFrame
    regime_series: pd.DataFrame


def _pick_winner(strategy_metrics: Dict[str, Metrics], rank_metric: str) -> Optional[str]:
    best_name = None
    best_value = float("-inf")
    for name, metrics in strategy_metrics.items():
        value = getattr(metrics, rank_metric)
        if pd.isna(value):
            continue
        if value > best_value:
            best_value = value
            best_name = name
    return best_name


def run_walk_forward(
    symbol: str,
    interval: str = "1h",
    regime_days: int = 20,
    test_days: int = 20,
    strategies: Optional[List[Strategy]] = None,
    engine_config: Optional[EngineConfig] = None,
    rank_metric: str = "sharpe_ratio",
    refresh: bool = True,
) -> WalkForwardReport:
    # Need at least 2 non-overlapping test windows (step=test_days) to say
    # anything about regime patterns -- get_bars raises InsufficientHistoryError
    # with a clear message if there isn't enough cached/fetched history.
    min_days = regime_days + test_days * 2
    bars = get_bars(symbol, interval=interval, refresh=refresh, min_days=min_days)

    trading_days = sorted(pd.unique(bars.index.date))
    strategies = strategies if strategies is not None else get_registered_strategies()
    engine = BacktestEngine(engine_config or EngineConfig())
    regime_series = classify_regime(bars)

    folds: List[WalkForwardFold] = []
    fold_id = 0
    i = 0
    while i + regime_days + test_days <= len(trading_days):
        regime_day_slice = trading_days[i:i + regime_days]
        test_day_slice = trading_days[i + regime_days:i + regime_days + test_days]
        regime_end_day = regime_day_slice[-1]

        regime_end_bars = bars[bars.index.date == regime_end_day]
        test_mask = (bars.index.date >= test_day_slice[0]) & (bars.index.date <= test_day_slice[-1])
        test_bars = bars[test_mask]

        if regime_end_bars.empty or test_bars.empty:
            i += test_days
            continue

        last_ts = regime_end_bars.index[-1]
        trend_regime = str(regime_series.loc[last_ts, "trend_regime"])
        vol_regime = str(regime_series.loc[last_ts, "vol_regime"])

        strategy_metrics: Dict[str, Metrics] = {}
        for strategy in strategies:
            try:
                result = engine.run(strategy, test_bars, symbol)
                strategy_metrics[strategy.name] = compute_metrics(result)
            except Exception:
                continue

        folds.append(WalkForwardFold(
            fold_id=fold_id,
            regime_start=regime_day_slice[0],
            regime_end=regime_end_day,
            test_start=test_day_slice[0],
            test_end=test_day_slice[-1],
            trend_regime=trend_regime,
            vol_regime=vol_regime,
            strategy_metrics=strategy_metrics,
            winner=_pick_winner(strategy_metrics, rank_metric),
        ))
        fold_id += 1
        i += test_days

    regime_strategy_map = _aggregate_regime_map(folds, rank_metric)

    last_bar_ts = bars.index[-1]
    current_regime = (
        str(regime_series.loc[last_bar_ts, "trend_regime"]),
        str(regime_series.loc[last_bar_ts, "vol_regime"]),
    )
    current_recommendation = _recommend(regime_strategy_map, current_regime, folds, rank_metric)

    return WalkForwardReport(
        symbol=symbol.upper(),
        interval=interval,
        rank_metric=rank_metric,
        folds=folds,
        regime_strategy_map=regime_strategy_map,
        current_regime=current_regime,
        current_recommendation=current_recommendation,
        bars=bars,
        regime_series=regime_series,
    )


def _aggregate_regime_map(
    folds: List[WalkForwardFold], rank_metric: str
) -> Dict[Tuple[str, str], Dict[str, dict]]:
    buckets: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for fold in folds:
        key = (fold.trend_regime, fold.vol_regime)
        bucket = buckets.setdefault(key, {})
        for name, metrics in fold.strategy_metrics.items():
            entry = bucket.setdefault(name, {"wins": 0, "folds": 0, "metric_sum": 0.0, "metric_count": 0})
            entry["folds"] += 1
            value = getattr(metrics, rank_metric)
            if not pd.isna(value):
                entry["metric_sum"] += value
                entry["metric_count"] += 1
        if fold.winner:
            bucket.setdefault(fold.winner, {"wins": 0, "folds": 0, "metric_sum": 0.0, "metric_count": 0})
            bucket[fold.winner]["wins"] += 1

    result: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for key, bucket in buckets.items():
        result[key] = {
            name: {
                "wins": entry["wins"],
                "folds": entry["folds"],
                "avg_metric": (entry["metric_sum"] / entry["metric_count"]) if entry["metric_count"] else float("nan"),
            }
            for name, entry in bucket.items()
        }
    return result


def rank_bucket(bucket: Dict[str, dict]) -> List[Tuple[str, dict]]:
    """Rank a regime bucket's strategies best-first.

    Average metric is the primary key, win-count only a tie-break -- with
    as few as 4-11 folds per bucket, win-count is noisy (a strategy can
    "win" more folds by tiny margins while another has a much better
    average risk-adjusted return, as seen on AAPL's ranging/high_vol
    bucket: ma_crossover won 2/4 folds but averaged -2.13 Sharpe, while
    vwap_reversion won only 1/4 but averaged +2.93). Average metric is the
    more statistically meaningful signal at this sample size.
    """

    def _key(item):
        _, stats = item
        avg_metric = stats["avg_metric"]
        return (avg_metric if not pd.isna(avg_metric) else float("-inf"), stats["wins"])

    return sorted(bucket.items(), key=_key, reverse=True)


def _best_in_bucket(bucket: Dict[str, dict]) -> Optional[str]:
    ranked = rank_bucket(bucket)
    return ranked[0][0] if ranked else None


def _recommend(
    regime_strategy_map: Dict[Tuple[str, str], Dict[str, dict]],
    current_regime: Tuple[str, str],
    folds: List[WalkForwardFold],
    rank_metric: str,
) -> Optional[str]:
    bucket = regime_strategy_map.get(current_regime)
    if bucket:
        return _best_in_bucket(bucket)

    # No historical folds for this exact regime -- fall back to the overall
    # best strategy computed directly across all folds.
    values_by_strategy: Dict[str, List[float]] = {}
    wins: Dict[str, int] = {}
    for fold in folds:
        for name, metrics in fold.strategy_metrics.items():
            value = getattr(metrics, rank_metric)
            if not pd.isna(value):
                values_by_strategy.setdefault(name, []).append(value)
        if fold.winner:
            wins[fold.winner] = wins.get(fold.winner, 0) + 1

    summary = {
        name: {
            "wins": wins.get(name, 0),
            "folds": len(values),
            "avg_metric": (sum(values) / len(values)) if values else float("nan"),
        }
        for name, values in values_by_strategy.items()
    }
    return _best_in_bucket(summary)
