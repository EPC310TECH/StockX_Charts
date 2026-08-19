from pathlib import Path
from typing import Optional

import pandas as pd

from stockx.config import CACHE_DIR, DEFAULT_MIN_HISTORY_DAYS
from stockx.data.fetch import fetch_max_available
from stockx.exceptions import InsufficientHistoryError

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _drop_bad_rows(bars: pd.DataFrame) -> pd.DataFrame:
    """yfinance occasionally reports a still-forming or otherwise-blank bar
    (real volume, but open/high/low/close all NaN -- seen on the current
    day's row for some symbols) that Python's json.dumps happily emits as a
    bare `NaN` token, which isn't valid JSON and breaks every downstream
    consumer of get_bars: the chart fails to load at all (the browser's
    strict JSON.parse rejects the whole response), and pattern/indicator/
    backtest math would silently propagate the NaN otherwise."""
    if bars.empty:
        return bars
    return bars.dropna(subset=[c for c in OHLCV_COLUMNS if c in bars.columns])


def cache_path(symbol: str, interval: str = "1m") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.upper()}_{interval}.parquet"


def load_cached(symbol: str, interval: str = "1m") -> pd.DataFrame:
    path = cache_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return pd.read_parquet(path)


def save_cached(symbol: str, new_bars: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    existing = load_cached(symbol, interval)
    # Concatenating with an empty (object-dtype) frame silently upcasts
    # numeric columns to object, which later breaks things like cumsum() in
    # indicators -- skip empty frames instead of concatenating them.
    frames = [df for df in (existing, new_bars) if not df.empty]
    merged = pd.concat(frames) if frames else new_bars
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = _drop_bad_rows(merged)
    merged.to_parquet(cache_path(symbol, interval))
    return merged


def get_bars(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1m",
    refresh: bool = True,
    min_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    """Load bars for `symbol`, merging in freshly fetched data when `refresh`
    is set. This is the mechanism by which local history accumulates beyond
    what any single yfinance call can return: repeated runs (e.g. daily or
    weekly) each pull the current ~30-day window and merge it into the
    on-disk cache, growing the effective backtest range over time."""
    symbol = symbol.upper()
    bars = _drop_bad_rows(load_cached(symbol, interval))

    if refresh:
        fresh = fetch_max_available(symbol, interval)
        bars = save_cached(symbol, fresh, interval)

    if bars.empty:
        raise InsufficientHistoryError(symbol, have_days=0, need_days=min_days)

    if start is not None or end is not None:
        bars = bars.loc[start:end]

    have_days = len(pd.unique(bars.index.date)) if not bars.empty else 0
    if have_days < min_days:
        raise InsufficientHistoryError(symbol, have_days=have_days, need_days=min_days)

    return bars
