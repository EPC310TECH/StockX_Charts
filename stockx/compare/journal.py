import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from stockx.config import REPORTS_DIR

# Single-user trade journal -- same storage idiom as watchlist.py/layouts.py
# (a flat JSON file under REPORTS_DIR, no auth, no database) rather than
# the multi-user Postgres+Supabase-Auth design this was adapted from. This
# logs trades *you* actually took (or are validating before taking), and
# is deliberately separate from the backtest engine's Trade dataclass
# (stockx/backtest/engine.py) -- that one records simulated fills from a
# historical strategy run; this one records real, manually-entered trades
# with a pre-entry checklist gate.

TRADES_JSON_PATH = REPORTS_DIR / "journal_trades.json"
SETTINGS_JSON_PATH = REPORTS_DIR / "journal_settings.json"

DEFAULT_SETTINGS = {
    "account_balance": 10_000.0,
    "risk_percentage": 2.0,
    "preferred_timeframes": ["5m"],
    "min_rr_ratio": 1.5,
}

OPEN = "open"
CLOSED_WIN = "closed_win"
CLOSED_LOSS = "closed_loss"
INVALIDATED = "invalidated"


def _load_json(path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_settings() -> dict:
    return {**DEFAULT_SETTINGS, **_load_json(SETTINGS_JSON_PATH, {})}


def save_settings(updates: dict) -> dict:
    settings = {**load_settings(), **updates}
    _save_json(SETTINGS_JSON_PATH, settings)
    return settings


def load_trades() -> List[dict]:
    return _load_json(TRADES_JSON_PATH, [])


def _save_trades(trades: List[dict]) -> None:
    _save_json(TRADES_JSON_PATH, trades)


def validate_trade_setup(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    side: str = "long",
    min_rr_ratio: float = 1.5,
    volume_confirmed: bool = False,
    higher_tf_aligned: bool = False,
) -> dict:
    """Pre-trade checklist gate: computes RR ratio from entry/stop/target
    and flags hard errors (bad stop/target placement, RR below minimum --
    these should block entry) separately from soft warnings (volume/
    higher-timeframe confirmation missing -- worth seeing, not blocking).
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    risk = (entry_price - stop_loss) if side == "long" else (stop_loss - entry_price)
    reward = (target_price - entry_price) if side == "long" else (entry_price - target_price)

    errors: List[str] = []
    warnings: List[str] = []
    rr_ratio: Optional[float] = None

    if risk <= 0:
        errors.append(
            "Stop loss must be on the risk side of entry "
            f"({'below' if side == 'long' else 'above'} entry for a {side} trade)"
        )
    elif reward <= 0:
        errors.append(
            "Target must be on the profit side of entry "
            f"({'above' if side == 'long' else 'below'} entry for a {side} trade)"
        )
    else:
        rr_ratio = reward / risk
        if rr_ratio < min_rr_ratio:
            errors.append(f"Risk/reward {rr_ratio:.2f} is below the minimum {min_rr_ratio}")

    if not volume_confirmed:
        warnings.append("Volume not confirmed")
    if not higher_tf_aligned:
        warnings.append("Higher timeframe not aligned")

    return {
        "is_valid": len(errors) == 0,
        "rr_ratio": rr_ratio,
        "errors": errors,
        "warnings": warnings,
    }


def suggested_quantity(account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float, side: str = "long") -> int:
    """Position size such that a full stop-out loses exactly
    `risk_percentage` of `account_balance` -- not a recommendation to take
    the trade, just the sizing math once you've decided to."""
    risk_per_share = (entry_price - stop_loss) if side == "long" else (stop_loss - entry_price)
    if risk_per_share <= 0:
        return 0
    max_risk_dollars = account_balance * (risk_percentage / 100.0)
    return int(max_risk_dollars // risk_per_share)


def create_trade(
    symbol: str,
    setup_type: str,
    side: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    quantity: int,
    volume_confirmed: bool = False,
    higher_tf_aligned: bool = False,
    notes: str = "",
) -> dict:
    settings = load_settings()
    validation = validate_trade_setup(
        entry_price, stop_loss, target_price, side, settings["min_rr_ratio"], volume_confirmed, higher_tf_aligned
    )
    trade = {
        "id": str(uuid.uuid4()),
        "symbol": symbol.upper(),
        "setup_type": setup_type,
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "quantity": quantity,
        "volume_confirmed": volume_confirmed,
        "higher_tf_aligned": higher_tf_aligned,
        "rr_ratio": validation["rr_ratio"],
        "exit_price": None,
        "fees": 0.0,
        "gross_pl": None,
        "net_pl": None,
        "status": OPEN,
        "entry_timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_timestamp": None,
        "notes": notes,
    }
    trades = load_trades()
    trades.append(trade)
    _save_trades(trades)
    return trade


def close_trade(trade_id: str, exit_price: float, fees: float = 0.0) -> dict:
    trades = load_trades()
    for trade in trades:
        if trade["id"] == trade_id:
            side_sign = 1 if trade["side"] == "long" else -1
            gross_pl = side_sign * (exit_price - trade["entry_price"]) * trade["quantity"]
            net_pl = gross_pl - fees
            trade["exit_price"] = exit_price
            trade["fees"] = fees
            trade["gross_pl"] = gross_pl
            trade["net_pl"] = net_pl
            trade["status"] = CLOSED_WIN if net_pl > 0 else CLOSED_LOSS
            trade["exit_timestamp"] = datetime.now(timezone.utc).isoformat()
            _save_trades(trades)
            return trade
    raise KeyError(f"no trade with id {trade_id!r}")


def update_trade_notes(trade_id: str, notes: str) -> dict:
    trades = load_trades()
    for trade in trades:
        if trade["id"] == trade_id:
            trade["notes"] = notes
            _save_trades(trades)
            return trade
    raise KeyError(f"no trade with id {trade_id!r}")


def invalidate_trade(trade_id: str) -> dict:
    trades = load_trades()
    for trade in trades:
        if trade["id"] == trade_id:
            trade["status"] = INVALIDATED
            _save_trades(trades)
            return trade
    raise KeyError(f"no trade with id {trade_id!r}")


def delete_trade(trade_id: str) -> List[dict]:
    trades = [t for t in load_trades() if t["id"] != trade_id]
    _save_trades(trades)
    return trades


def _setup_bucket_stats(trades: List[dict]) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = {}
    for t in trades:
        buckets.setdefault(t["setup_type"], []).append(t)

    result: Dict[str, dict] = {}
    for setup_type, group in buckets.items():
        result[setup_type] = _bucket_stats(group)
    return result


def _bucket_stats(trades: List[dict]) -> dict:
    wins = [t["net_pl"] for t in trades if t["net_pl"] > 0]
    losses = [t["net_pl"] for t in trades if t["net_pl"] <= 0]
    n = len(trades)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "total_trades": n,
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate": (len(wins) / n) if n else None,
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (-gross_loss / len(losses)) if losses else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else None),
        "net_pl": sum(t["net_pl"] for t in trades),
    }


def filter_trades(
    symbol: Optional[str] = None,
    setup_type: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[dict]:
    trades = load_trades()
    if symbol:
        trades = [t for t in trades if t["symbol"] == symbol.upper()]
    if setup_type:
        trades = [t for t in trades if t["setup_type"] == setup_type]
    if status:
        trades = [t for t in trades if t["status"] == status]
    if from_date:
        trades = [t for t in trades if t["entry_timestamp"][:10] >= from_date]
    if to_date:
        trades = [t for t in trades if t["entry_timestamp"][:10] <= to_date]
    return sorted(trades, key=lambda t: t["entry_timestamp"], reverse=True)


def compute_journal_stats(trades: Optional[List[dict]] = None) -> dict:
    """Win rate / profit factor / avg win-loss, overall and broken down by
    setup_type -- only over CLOSED trades (open/invalidated trades have no
    realized P&L to score). Mirrors compute_pattern_stats' reliability
    framing, applied to trades you actually took instead of backtested
    signals."""
    trades = trades if trades is not None else load_trades()
    closed = [t for t in trades if t["status"] in (CLOSED_WIN, CLOSED_LOSS) and t.get("net_pl") is not None]

    return {
        "overall": _bucket_stats(closed) if closed else _bucket_stats([]),
        "by_setup_type": _setup_bucket_stats(closed),
    }
