import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from stockx.backtest.engine import BacktestResult
from stockx.config import PERIOD_CONTEXT_BARS

# Below this equity change, a "period" isn't a real move -- e.g. a strategy
# that never traded would otherwise report a zero-length, zero-change
# best/worst period.
MIN_EQUITY_CHANGE = 1e-6


@dataclass
class WindowStats:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    price_change_pct: float
    avg_volume: float
    volume_ratio: float
    volatility: float
    num_bars: int


@dataclass
class PeriodEvent:
    kind: str  # "best" | "worst"
    start: pd.Timestamp
    end: pd.Timestamp
    equity_change: float
    equity_change_pct: float
    pre: WindowStats
    during: WindowStats
    post: WindowStats


def _describe_window(segment: pd.DataFrame, overall_avg_volume: float) -> WindowStats:
    if segment.empty:
        return WindowStats(None, None, float("nan"), float("nan"), float("nan"), float("nan"), 0)

    first_close = segment["close"].iloc[0]
    price_change_pct = (segment["close"].iloc[-1] / first_close - 1) if first_close else float("nan")
    avg_volume = float(segment["volume"].mean())
    volume_ratio = (avg_volume / overall_avg_volume) if overall_avg_volume else float("nan")

    bar_returns = segment["close"].pct_change().dropna()
    volatility = float(bar_returns.std()) if len(bar_returns) > 1 else float("nan")

    return WindowStats(
        start=segment.index[0],
        end=segment.index[-1],
        price_change_pct=float(price_change_pct),
        avg_volume=avg_volume,
        volume_ratio=float(volume_ratio),
        volatility=volatility,
        num_bars=len(segment),
    )


def _build_event(
    kind: str, start: pd.Timestamp, end: pd.Timestamp, bars: pd.DataFrame, equity: pd.Series
) -> Optional[PeriodEvent]:
    change = float(equity.loc[end] - equity.loc[start])
    if abs(change) < MIN_EQUITY_CHANGE:
        return None
    change_pct = (change / equity.loc[start]) if equity.loc[start] else float("nan")

    idx = bars.index
    start_i = idx.get_loc(start)
    end_i = idx.get_loc(end)
    overall_avg_volume = float(bars["volume"].mean())

    pre_seg = bars.iloc[max(0, start_i - PERIOD_CONTEXT_BARS):start_i]
    during_seg = bars.iloc[start_i:end_i + 1]
    post_seg = bars.iloc[end_i + 1:end_i + 1 + PERIOD_CONTEXT_BARS]

    return PeriodEvent(
        kind=kind,
        start=start,
        end=end,
        equity_change=change,
        equity_change_pct=float(change_pct),
        pre=_describe_window(pre_seg, overall_avg_volume),
        during=_describe_window(during_seg, overall_avg_volume),
        post=_describe_window(post_seg, overall_avg_volume),
    )


def find_best_worst_periods(
    result: BacktestResult, bars: pd.DataFrame
) -> Tuple[Optional[PeriodEvent], Optional[PeriodEvent]]:
    """Identify each strategy's single largest run-up ('best') and largest
    drawdown ('worst') period from its equity curve, using the standard
    peak/trough decomposition, then describe market conditions in `bars`
    just before, during, and just after each period."""
    equity = result.equity_curve
    if len(equity) < 2:
        return None, None

    running_max = equity.cummax()
    drawdown = equity - running_max
    trough_idx = drawdown.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    worst = _build_event("worst", peak_idx, trough_idx, bars, equity)

    running_min = equity.cummin()
    runup = equity - running_min
    peak_idx2 = runup.idxmax()
    trough_idx2 = equity.loc[:peak_idx2].idxmin()
    best = _build_event("best", trough_idx2, peak_idx2, bars, equity)

    return best, worst


def _fmt_pct(value: float) -> str:
    return f"{value:+.2%}" if not math.isnan(value) else "N/A"


def _fmt_ratio(value: float) -> str:
    return f"{value:.2f}" if not math.isnan(value) else "N/A"


def _fmt_num(value: float) -> str:
    return f"{value:.4f}" if not math.isnan(value) else "N/A"


def format_period_bullets(event: Optional[PeriodEvent], label: str) -> List[str]:
    if event is None:
        return [f"{label}: no distinct period identified (insufficient trading activity)."]

    return [
        f"{label} period: {event.start} -> {event.end}  "
        f"({event.equity_change:+,.2f}, {event.equity_change_pct:+.2%})",
        f"  Before ({event.pre.num_bars} bars): price {_fmt_pct(event.pre.price_change_pct)}, "
        f"volume {_fmt_ratio(event.pre.volume_ratio)}x avg, volatility {_fmt_num(event.pre.volatility)}",
        f"  During ({event.during.num_bars} bars): price {_fmt_pct(event.during.price_change_pct)}, "
        f"volume {_fmt_ratio(event.during.volume_ratio)}x avg, volatility {_fmt_num(event.during.volatility)}",
        f"  After  ({event.post.num_bars} bars): price {_fmt_pct(event.post.price_change_pct)}, "
        f"volume {_fmt_ratio(event.post.volume_ratio)}x avg, volatility {_fmt_num(event.post.volatility)}",
    ]
