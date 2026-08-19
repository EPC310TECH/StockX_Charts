from datetime import datetime

import pandas as pd

from stockx.config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, DATABENTO_API_KEY, EXCHANGE_TZ
from stockx.exceptions import DataFetchError, MissingCredentialsError

TICK_SOURCES = ("alpaca", "databento")

TRADE_COLUMNS = ["price", "size", "exchange", "conditions"]
QUOTE_COLUMNS = ["bid_price", "bid_size", "ask_price", "ask_size", "bid_exchange", "ask_exchange"]

# Databento's consolidated US equities feed -- the closest match to a
# full-tape SIP-style view available at the "basic" price tier, as opposed
# to single-venue datasets like XNAS.ITCH (Nasdaq only).
DATABENTO_DEFAULT_DATASET = "DBEQ.BASIC"


def _require_alpaca_credentials() -> None:
    missing = [
        name
        for name, value in (("ALPACA_API_KEY_ID", ALPACA_API_KEY_ID), ("ALPACA_API_SECRET_KEY", ALPACA_API_SECRET_KEY))
        if not value
    ]
    if missing:
        raise MissingCredentialsError("Alpaca", missing)


def _require_databento_credentials() -> None:
    if not DATABENTO_API_KEY:
        raise MissingCredentialsError("Databento", ["DATABENTO_API_KEY"])


def _alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient

    _require_alpaca_credentials()
    return StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)


def _normalize_alpaca_df(df: pd.DataFrame, symbol: str, columns: list) -> pd.DataFrame:
    """Alpaca's `.df` comes back with a (symbol, timestamp) MultiIndex
    (built to hold multiple symbols per response) -- collapse to a single
    tz-aware timestamp index for one symbol, matching the shape
    `stockx.data.cache` bars already use elsewhere in this codebase."""
    if df.empty:
        return pd.DataFrame(columns=columns)
    if "symbol" in df.index.names:
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EXCHANGE_TZ)
    df.index.name = "timestamp"
    keep = [c for c in columns if c in df.columns]
    df = df[keep].sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_alpaca_trades(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Free-tier tick-level trades via Alpaca's IEX feed -- roughly 2.5% of
    consolidated US equity volume (one exchange, not the full tape), but
    genuine tick data at no cost. This is the default tick source."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockTradesRequest

    client = _alpaca_client()
    request = StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=DataFeed.IEX)
    try:
        trade_set = client.get_stock_trades(request)
    except Exception as exc:
        raise DataFetchError(symbol, f"Alpaca trades request failed: {exc}") from exc
    return _normalize_alpaca_df(trade_set.df, symbol, TRADE_COLUMNS)


def fetch_alpaca_quotes(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Free-tier tick-level NBBO-style quotes via Alpaca's IEX feed (same
    single-venue caveat as fetch_alpaca_trades)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest

    client = _alpaca_client()
    request = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=DataFeed.IEX)
    try:
        quote_set = client.get_stock_quotes(request)
    except Exception as exc:
        raise DataFetchError(symbol, f"Alpaca quotes request failed: {exc}") from exc
    return _normalize_alpaca_df(quote_set.df, symbol, QUOTE_COLUMNS)


def fetch_databento_trades(
    symbol: str, start: datetime, end: datetime, dataset: str = DATABENTO_DEFAULT_DATASET
) -> pd.DataFrame:
    """Premium full-tape trades, billed against the account's Databento
    credit. Only call this when full-tape accuracy actually matters for
    the task at hand (e.g. spot-checking an Alpaca/IEX-derived result) --
    not as a routine, repeatedly-called data source."""
    import databento as db

    _require_databento_credentials()
    client = db.Historical(DATABENTO_API_KEY)
    try:
        store = client.timeseries.get_range(
            dataset=dataset, symbols=symbol, schema="trades", start=start, end=end
        )
        df = store.to_df(tz=EXCHANGE_TZ)
    except Exception as exc:
        raise DataFetchError(symbol, f"Databento trades request failed: {exc}") from exc
    if df.empty:
        return pd.DataFrame(columns=["price", "size", "side"])
    df.index.name = "timestamp"
    keep = [c for c in ("price", "size", "side", "publisher_id") if c in df.columns]
    return df[keep].sort_index()


def get_ticks(
    symbol: str, start: datetime, end: datetime, source: str = "alpaca", kind: str = "trades"
) -> pd.DataFrame:
    """Unified entry point. `source='alpaca'` (default, free, IEX-only) or
    `source='databento'` (opt-in, spends the account's finite credit) --
    the two never silently fall back to each other, since that could spend
    premium credit on a call the caller expected to be free."""
    if source == "alpaca":
        if kind == "trades":
            return fetch_alpaca_trades(symbol, start, end)
        if kind == "quotes":
            return fetch_alpaca_quotes(symbol, start, end)
        raise ValueError(f"unknown kind {kind!r}; alpaca supports 'trades' or 'quotes'")
    if source == "databento":
        if kind != "trades":
            raise ValueError("databento source currently only supports kind='trades'")
        return fetch_databento_trades(symbol, start, end)
    raise ValueError(f"unknown source {source!r}; supported: {TICK_SOURCES}")
