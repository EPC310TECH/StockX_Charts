from typing import List

import pandas as pd

from stockx.backtest.engine import BacktestEngine, EngineConfig
from stockx.compare.orchestrator import run_comparison
from stockx.data.cache import get_bars
from stockx.metrics.metrics import Metrics, compute_metrics
from stockx.strategies.base import STRATEGY_REGISTRY

# Live symbol lookups already fetch fresh bars for the chart itself
# (stockx.server.api_symbol); backtests triggered from the dashboard reuse
# that same cached data rather than paying a second live yfinance fetch.
_REFRESH = False


def _safe(value):
    return None if isinstance(value, float) and pd.isna(value) else value


def _metrics_to_dict(m: Metrics) -> dict:
    return {
        "total_return": _safe(m.total_return),
        "annualized_return": _safe(m.annualized_return),
        "sharpe_ratio": _safe(m.sharpe_ratio),
        "sortino_ratio": _safe(m.sortino_ratio),
        "max_drawdown": _safe(m.max_drawdown),
        "win_rate": _safe(m.win_rate),
        "profit_factor": _safe(m.profit_factor),
        "num_trades": m.num_trades,
        "avg_trade_return_pct": _safe(m.avg_trade_return_pct),
    }


def _equity_curve_to_points(equity: pd.Series) -> List[dict]:
    return [{"time": int(ts.timestamp()), "value": float(v)} for ts, v in equity.items()]


def _trades_to_rows(trades) -> List[dict]:
    """One row per round-trip trade, carrying every Trade field -- the
    single source both the chart markers (2 points per trade, derived
    client-side) and the trades table/CSV export are built from, so there's
    one shape to keep in sync rather than two."""
    return [
        {
            "entry_time": int(t.entry_time.timestamp()),
            "entry_price": t.entry_price,
            "exit_time": int(t.exit_time.timestamp()),
            "exit_price": t.exit_price,
            "side": t.side,
            "qty": t.qty,
            "pnl": t.pnl,
            "return_pct": t.return_pct,
            "commission": t.commission,
            "slippage": t.slippage,
        }
        for t in trades
    ]


def list_strategies() -> List[dict]:
    return [
        {
            "name": name,
            "display_name": cls.display_name,
            "default_params": cls().default_params(),
            "param_choices": cls().param_choices(),
        }
        for name, cls in STRATEGY_REGISTRY.items()
    ]


def _coerce_params(strategy_cls, raw_params: dict) -> dict:
    """JSON numbers round-trip through JS as whatever the input's step/type
    produced -- coerce back to the type each param's own default declares
    (e.g. `fast`/`period` are rolling-window sizes and must land as int, not
    float, or pandas' .rolling() rejects them) rather than trusting the
    client."""
    defaults = strategy_cls().default_params()
    coerced = {}
    for key, value in raw_params.items():
        if key not in defaults:
            continue
        default_value = defaults[key]
        if isinstance(default_value, bool):
            coerced[key] = bool(value)
        elif isinstance(default_value, int):
            coerced[key] = int(value)
        elif isinstance(default_value, float):
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return coerced


EXECUTION_TIMING_OPTIONS = ("next_open", "same_close")
INTRABAR_PATH_OPTIONS = ("ohlc", "olhc")


def run_single_backtest(
    symbol: str,
    interval: str,
    strategy_name: str,
    params: dict,
    execution_timing: str = "next_open",
    intrabar_path: str = "ohlc",
) -> dict:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy {strategy_name!r}")
    if execution_timing not in EXECUTION_TIMING_OPTIONS:
        raise ValueError(f"unknown execution_timing {execution_timing!r}; options: {EXECUTION_TIMING_OPTIONS}")
    if intrabar_path not in INTRABAR_PATH_OPTIONS:
        raise ValueError(f"unknown intrabar_path {intrabar_path!r}; options: {INTRABAR_PATH_OPTIONS}")
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    strategy = strategy_cls(**_coerce_params(strategy_cls, params or {}))

    bars = get_bars(symbol, interval=interval, refresh=_REFRESH)
    engine = BacktestEngine(EngineConfig(execution_timing=execution_timing, intrabar_path=intrabar_path))
    result = engine.run(strategy, bars, symbol.upper())
    metrics = compute_metrics(result)

    return {
        "symbol": result.symbol,
        "strategy": result.strategy_name,
        "display_name": strategy_cls.display_name,
        "params": result.params,
        "execution_timing": execution_timing,
        "intrabar_path": intrabar_path,
        "initial_capital": result.initial_capital,
        "equity_curve": _equity_curve_to_points(result.equity_curve),
        "trades": _trades_to_rows(result.trades),
        "metrics": _metrics_to_dict(metrics),
    }


def run_strategy_comparison(symbol: str, interval: str) -> dict:
    report = run_comparison(symbol, interval=interval, refresh=_REFRESH)
    display_names = {name: cls.display_name for name, cls in STRATEGY_REGISTRY.items()}

    results = [
        {
            "strategy": result.strategy_name,
            "display_name": display_names.get(result.strategy_name, result.strategy_name),
            "metrics": _metrics_to_dict(metrics),
        }
        for result, metrics in report.results
    ]
    benchmark = None
    if report.benchmark_result is not None and report.benchmark_metrics is not None:
        benchmark = {
            "strategy": report.benchmark_result.strategy_name,
            "metrics": _metrics_to_dict(report.benchmark_metrics),
        }

    return {
        "symbol": report.symbol,
        "interval": report.interval,
        "rank_metric": report.rank_metric,
        "best_strategy": report.best_strategy,
        "results": results,
        "benchmark": benchmark,
        "failed": [{"strategy": name, "error": err} for name, err in report.failed],
    }
