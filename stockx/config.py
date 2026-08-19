import os
from pathlib import Path

from dotenv import load_dotenv

# Loads a local .env (if present) into os.environ without overwriting
# variables the shell already exported -- lets API keys live in a
# gitignored file instead of the shell profile, same as any other secret.
load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
TICK_CACHE_DIR = Path(__file__).resolve().parent.parent / "tick_cache"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Tick/quote data credentials (see .env.example). Alpaca is the free-tier
# default (IEX feed); Databento is the opt-in premium source, billed
# against a finite credit, so it's only ever called when a caller
# explicitly asks for it -- never a silent fallback.
ALPACA_API_KEY_ID = os.environ.get("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.environ.get("ALPACA_API_SECRET_KEY")
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY")

EXCHANGE_TZ = "America/New_York"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

TRADING_DAYS_PER_YEAR = 252

DEFAULT_MIN_HISTORY_DAYS = 5

# Number of bars of context shown before/after a best/worst equity period
# in the PDF report, to describe what conditions led into and out of it.
PERIOD_CONTEXT_BARS = 10

# yfinance's actual intraday retention/request-size limits per interval:
# 1m keeps 30 days of history with a 7-day cap per request; 5m/15m/30m keep
# 60 days; 1h (=60m) keeps ~730 days. Daily bars have no such retention
# limit (yfinance serves decades of history in one request), so a single
# ~10-year chunk covers it. Note: EOD-flatten, session VWAP, and ORB are
# intraday-session concepts that lose their meaning at daily granularity --
# fine for the candlestick/chart-pattern dashboard (timeframe-agnostic),
# but the full strategy-comparison engine's session-based strategies
# shouldn't be trusted at "1d".
INTERVAL_LIMITS = {
    "1m": {"max_lookback_days": 30, "max_chunk_days": 7},
    "5m": {"max_lookback_days": 60, "max_chunk_days": 60},
    "15m": {"max_lookback_days": 60, "max_chunk_days": 60},
    "30m": {"max_lookback_days": 60, "max_chunk_days": 60},
    # 725, not 730: yfinance's actual boundary check rejects a request
    # spanning exactly 730 days (off-by-one on their end), so this leaves a
    # small safety margin rather than tripping "must be within the last
    # 730 days" on every fresh fetch.
    "1h": {"max_lookback_days": 725, "max_chunk_days": 725},
    "1d": {"max_lookback_days": 3650, "max_chunk_days": 3650},
}
SUPPORTED_INTERVALS = list(INTERVAL_LIMITS.keys())
