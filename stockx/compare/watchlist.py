import json
from typing import List

from stockx.config import REPORTS_DIR

# Plain ordered list of symbols -- no interval, since interval stays a
# global chart control rather than something tracked per watchlist entry.
WATCHLIST_JSON_PATH = REPORTS_DIR / "watchlist.json"


def load_watchlist() -> List[str]:
    if not WATCHLIST_JSON_PATH.exists():
        return []
    with open(WATCHLIST_JSON_PATH) as f:
        return json.load(f)


def _save_watchlist(symbols: List[str]) -> None:
    WATCHLIST_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_JSON_PATH, "w") as f:
        json.dump(symbols, f)


def add_symbol(symbol: str) -> List[str]:
    symbol = symbol.upper()
    symbols = load_watchlist()
    if symbol not in symbols:
        symbols.append(symbol)
        _save_watchlist(symbols)
    return symbols


def remove_symbol(symbol: str) -> List[str]:
    symbol = symbol.upper()
    symbols = [s for s in load_watchlist() if s != symbol]
    _save_watchlist(symbols)
    return symbols
