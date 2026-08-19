import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockx.backtest.engine import BacktestResult
from stockx.config import TRADING_DAYS_PER_YEAR


@dataclass
class Metrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return_pct: float


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.0) -> Metrics:
    equity = result.equity_curve

    num_days = len(pd.unique(equity.index.date))
    # Observed bars/day rather than a hardcoded 1-minute constant, so this
    # stays correct at any bar interval (5m, 15m, 30m, 1h, ...) and for
    # partial/half trading days, without needing an interval parameter.
    bars_per_day = (len(equity) / num_days) if num_days > 0 else 1.0
    bars_per_year = bars_per_day * TRADING_DAYS_PER_YEAR
    annualization_factor = math.sqrt(bars_per_year)
    rf_per_bar = risk_free_rate / bars_per_year

    bar_returns = equity.pct_change().dropna()

    total_return = (equity.iloc[-1] / result.initial_capital) - 1 if len(equity) else np.nan

    if num_days > 0 and (1 + total_return) > 0:
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / num_days) - 1
    else:
        annualized_return = np.nan

    sharpe_ratio = _sharpe(bar_returns, rf_per_bar, annualization_factor)
    sortino_ratio = _sortino(bar_returns, rf_per_bar, annualization_factor)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown = drawdown.min() if len(drawdown) else np.nan

    trades = result.trades
    num_trades = len(trades)
    if num_trades > 0:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / num_trades
        profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
        avg_trade_return_pct = float(np.mean([t.return_pct for t in trades]))
    else:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_return_pct = np.nan

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        avg_trade_return_pct=avg_trade_return_pct,
    )


def _sharpe(bar_returns: pd.Series, rf_per_bar: float, annualization_factor: float) -> float:
    if len(bar_returns) < 2:
        return np.nan
    excess = bar_returns - rf_per_bar
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess.mean() / std) * annualization_factor


def _sortino(bar_returns: pd.Series, rf_per_bar: float, annualization_factor: float) -> float:
    if len(bar_returns) < 2:
        return np.nan
    excess = bar_returns - rf_per_bar
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * annualization_factor
