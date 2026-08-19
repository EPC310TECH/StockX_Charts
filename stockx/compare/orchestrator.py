from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import pandas as pd

from stockx.backtest.engine import BacktestEngine, BacktestResult, EngineConfig
from stockx.data.cache import get_bars
from stockx.metrics.metrics import Metrics, compute_metrics
from stockx.strategies import get_registered_strategies
from stockx.strategies.base import Strategy
from stockx.strategies.buy_and_hold import BuyAndHold

RANK_METRICS = ("sharpe_ratio", "sortino_ratio")


@dataclass
class ComparisonReport:
    symbol: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp
    results: List[Tuple[BacktestResult, Metrics]]
    failed: List[Tuple[str, str]]
    best_strategy: Optional[str]
    rank_metric: str
    bars: pd.DataFrame
    benchmark_result: Optional[BacktestResult]
    benchmark_metrics: Optional[Metrics]


def run_comparison(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1m",
    strategies: Optional[List[Strategy]] = None,
    engine_config: Optional[EngineConfig] = None,
    rank_metric: str = "sharpe_ratio",
    refresh: bool = True,
) -> ComparisonReport:
    if rank_metric not in RANK_METRICS:
        raise ValueError(f"rank_metric must be one of {RANK_METRICS}, got {rank_metric!r}")

    bars = get_bars(symbol, start=start, end=end, interval=interval, refresh=refresh)
    strategies = strategies if strategies is not None else get_registered_strategies()
    base_engine_config = engine_config or EngineConfig()
    engine = BacktestEngine(base_engine_config)

    results: List[Tuple[BacktestResult, Metrics]] = []
    failed: List[Tuple[str, str]] = []

    for strategy in strategies:
        try:
            result = engine.run(strategy, bars, symbol)
            metrics = compute_metrics(result)
            results.append((result, metrics))
        except Exception as exc:
            failed.append((strategy.name, str(exc)))

    def _sort_key(item: Tuple[BacktestResult, Metrics]) -> float:
        value = getattr(item[1], rank_metric)
        # NaN sorts last regardless of direction
        return float("-inf") if pd.isna(value) else value

    results.sort(key=_sort_key, reverse=True)
    best_strategy = results[0][0].strategy_name if results else None

    # Buy & hold is a reference benchmark, not a ranked candidate: it needs
    # flatten_eod=False (must hold overnight/across days, unlike every
    # registered day-trading strategy) but otherwise uses the same starting
    # capital, commission, slippage, and sizing for an apples-to-apples
    # "same time frame, same set amount" comparison.
    benchmark_result: Optional[BacktestResult] = None
    benchmark_metrics: Optional[Metrics] = None
    try:
        benchmark_engine = BacktestEngine(replace(base_engine_config, flatten_eod=False))
        benchmark_result = benchmark_engine.run(BuyAndHold(), bars, symbol)
        benchmark_metrics = compute_metrics(benchmark_result)
    except Exception as exc:
        failed.append((BuyAndHold.name, str(exc)))

    return ComparisonReport(
        symbol=symbol.upper(),
        interval=interval,
        start=bars.index.min(),
        end=bars.index.max(),
        results=results,
        failed=failed,
        best_strategy=best_strategy,
        rank_metric=rank_metric,
        bars=bars,
        benchmark_result=benchmark_result,
        benchmark_metrics=benchmark_metrics,
    )
