from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from stockx.config import TICK_CACHE_DIR
from stockx.data.ticks import get_ticks

# Tick data runs orders of magnitude more rows per symbol-day than OHLCV
# bars, so unlike stockx.data.cache (one accumulating parquet per symbol),
# each calendar day gets its own file -- lets a range query reuse whichever
# days are already on disk without reading/rewriting unrelated history,
# and (for Databento especially, where every re-pull spends the account's
# finite credit) makes "already fetched" a cheap file-existence check.


def _partition_path(symbol: str, source: str, kind: str, day: date) -> Path:
    dir_path = TICK_CACHE_DIR / source / kind / symbol.upper()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{day.isoformat()}.parquet"


def _load_day(symbol: str, source: str, kind: str, day: date) -> Optional[pd.DataFrame]:
    path = _partition_path(symbol, source, kind, day)
    return pd.read_parquet(path) if path.exists() else None


def _save_day(symbol: str, source: str, kind: str, day: date, df: pd.DataFrame) -> None:
    df.to_parquet(_partition_path(symbol, source, kind, day))


def get_ticks_cached(
    symbol: str,
    start: date,
    end: date,
    source: str = "alpaca",
    kind: str = "trades",
    refresh_today: bool = True,
) -> pd.DataFrame:
    """Range query over the per-day tick cache, fetching only the days not
    already on disk. `start`/`end` are calendar dates (inclusive).

    A historical day, once cached, is treated as complete and never
    refetched -- trades/quotes for a finished trading day don't change.
    Today's date is the one exception (`refresh_today=True`, the default):
    since the session may still be in progress, it's always fetched fresh
    rather than trusting a possibly-incomplete cached copy.
    """
    symbol = symbol.upper()
    today = datetime.now().date()

    frames = []
    day = start
    while day <= end:
        is_today = day == today
        cached = None if (is_today and refresh_today) else _load_day(symbol, source, kind, day)
        if cached is not None:
            frames.append(cached)
        else:
            day_start = datetime.combine(day, datetime.min.time())
            day_end = datetime.combine(day, datetime.max.time())
            fetched = get_ticks(symbol, day_start, day_end, source=source, kind=kind)
            # Only persist a day once it's safely in the past -- caching a
            # partial "today" would otherwise get treated as that day's
            # complete history on every later call within this function.
            if not is_today:
                _save_day(symbol, source, kind, day, fetched)
            if not fetched.empty:
                frames.append(fetched)
        day += timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined
