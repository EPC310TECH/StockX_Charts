from datetime import datetime, timedelta, date
from typing import List

import pandas as pd
import yfinance as yf

from stockx.config import EXCHANGE_TZ, INTERVAL_LIMITS
from stockx.exceptions import DataFetchError

YF_COLUMN_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
OHLCV_COLUMNS = list(YF_COLUMN_MAP.values())


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataFetchError(symbol, "no data returned")
    df = df.rename(columns=YF_COLUMN_MAP)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataFetchError(symbol, f"response missing columns: {missing}")
    df = df[OHLCV_COLUMNS].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EXCHANGE_TZ)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_intraday_chunk(symbol: str, start: date, end: date, interval: str = "1m") -> pd.DataFrame:
    """Fetch a single date-range chunk from yfinance. `Ticker(...).history()`
    is used instead of `yf.download()` so single-symbol responses come back
    with flat (non-MultiIndex) columns."""
    ticker = yf.Ticker(symbol)
    try:
        df = ticker.history(start=start, end=end, interval=interval)
    except Exception as exc:
        raise DataFetchError(symbol, f"yfinance request failed: {exc}") from exc
    return _normalize(df, symbol)


def fetch_max_available(symbol: str, interval: str = "1m") -> pd.DataFrame:
    """Fetch as much intraday history as Yahoo currently retains for
    `interval`, chunked to respect yfinance's per-request window limits."""
    if interval not in INTERVAL_LIMITS:
        raise ValueError(f"unsupported interval {interval!r}; supported: {list(INTERVAL_LIMITS)}")
    limits = INTERVAL_LIMITS[interval]
    lookback_days = limits["max_lookback_days"]
    chunk_days = limits["max_chunk_days"]

    today = datetime.utcnow().date()
    window_start = today - timedelta(days=lookback_days)

    chunks: List[pd.DataFrame] = []
    chunk_start = window_start
    while chunk_start < today:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), today)
        try:
            chunks.append(fetch_intraday_chunk(symbol, chunk_start, chunk_end, interval))
        except DataFetchError:
            pass  # weekends/holidays/no-data gaps are expected within the window
        chunk_start = chunk_end

    if not chunks:
        raise DataFetchError(
            symbol, f"no {interval} data available in the last {lookback_days} days"
        )

    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined
