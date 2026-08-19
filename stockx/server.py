import math

from flask import Flask, jsonify, request

from stockx.analysis.scanner import scan_watchlist
from stockx.compare import journal
from stockx.compare.layouts import get_layout, save_layout
from stockx.compare.patterns_html import (
    _load_dashboard_data,
    _render_dashboard_html,
    _save_dashboard_data,
    compute_symbol_entry,
)
from stockx.compare.watchlist import add_symbol, load_watchlist, remove_symbol
from stockx.compare.web_backtest import list_strategies, run_single_backtest, run_strategy_comparison
from stockx.config import SUPPORTED_INTERVALS
from stockx.data.cache import get_bars
from stockx.exceptions import DataFetchError, InsufficientHistoryError

app = Flask(__name__)


def _sanitize_floats(obj):
    """Recursively swaps NaN/Infinity floats for None -- both are valid
    Python floats (e.g. a zero-loss bucket's profit_factor is genuinely
    infinite) but neither is valid JSON; json.dumps emits them as bare
    NaN/Infinity tokens that a browser's strict JSON.parse rejects,
    breaking the whole response (the exact class of bug already fixed
    once this session for NaN OHLC rows)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


@app.route("/")
def index():
    data = _load_dashboard_data()
    return _render_dashboard_html(data, load_watchlist())


@app.route("/api/symbol/<symbol>")
def api_symbol(symbol):
    interval = request.args.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400

    symbol = symbol.upper()
    try:
        bars = get_bars(symbol, interval=interval, refresh=True)
    except (DataFetchError, InsufficientHistoryError) as exc:
        return jsonify({"error": str(exc)}), 404

    entry = compute_symbol_entry(symbol, interval, bars, forward_bars=10)

    # Live lookups stick around, same as CLI runs -- persist into the
    # shared dashboard store so this symbol@interval is there next time too.
    data = _load_dashboard_data()
    data[f"{symbol}@{interval}"] = entry
    _save_dashboard_data(data)

    return jsonify(entry)


@app.route("/api/watchlist", methods=["POST"])
def api_watchlist_add():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).strip()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    return jsonify({"symbols": add_symbol(symbol)})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def api_watchlist_remove(symbol):
    return jsonify({"symbols": remove_symbol(symbol)})


@app.route("/api/layout/<symbol>", methods=["GET"])
def api_layout_get(symbol):
    interval = request.args.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400
    return jsonify(get_layout(symbol, interval))


@app.route("/api/layout/<symbol>", methods=["POST"])
def api_layout_save(symbol):
    body = request.get_json(silent=True) or {}
    interval = body.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400
    save_layout(symbol, interval, body.get("drawings", []), body.get("indicators", []), body.get("panes", []))
    return jsonify({"ok": True})


@app.route("/api/strategies")
def api_strategies():
    return jsonify(list_strategies())


@app.route("/api/backtest/<symbol>", methods=["POST"])
def api_backtest(symbol):
    body = request.get_json(silent=True) or {}
    interval = body.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400
    strategy_name = body.get("strategy")
    if not strategy_name:
        return jsonify({"error": "strategy is required"}), 400

    try:
        result = run_single_backtest(
            symbol,
            interval,
            strategy_name,
            body.get("params", {}),
            execution_timing=body.get("execution_timing", "next_open"),
            intrabar_path=body.get("intrabar_path", "ohlc"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (DataFetchError, InsufficientHistoryError) as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(result)


@app.route("/api/compare/<symbol>", methods=["POST"])
def api_compare(symbol):
    body = request.get_json(silent=True) or {}
    interval = body.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400

    try:
        result = run_strategy_comparison(symbol, interval)
    except (DataFetchError, InsufficientHistoryError) as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(result)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    body = request.get_json(silent=True) or {}
    interval = body.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"}), 400
    symbols = load_watchlist()
    if not symbols:
        return jsonify({"error": "Watchlist is empty -- star a symbol first."}), 400
    return jsonify(scan_watchlist(symbols, interval))


@app.route("/api/journal/settings", methods=["GET"])
def api_journal_settings_get():
    return jsonify(journal.load_settings())


@app.route("/api/journal/settings", methods=["POST"])
def api_journal_settings_save():
    body = request.get_json(silent=True) or {}
    return jsonify(journal.save_settings(body))


@app.route("/api/journal/validate", methods=["POST"])
def api_journal_validate():
    body = request.get_json(silent=True) or {}
    try:
        entry_price = float(body["entry_price"])
        stop_loss = float(body["stop_loss"])
        target_price = float(body["target_price"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "entry_price, stop_loss, and target_price are required numbers"}), 400
    side = body.get("side", "long")
    settings = journal.load_settings()
    result = journal.validate_trade_setup(
        entry_price, stop_loss, target_price, side,
        min_rr_ratio=settings["min_rr_ratio"],
        volume_confirmed=bool(body.get("volume_confirmed", False)),
        higher_tf_aligned=bool(body.get("higher_tf_aligned", False)),
    )
    if result["rr_ratio"] is not None:
        result["suggested_quantity"] = journal.suggested_quantity(
            settings["account_balance"], settings["risk_percentage"], entry_price, stop_loss, side
        )
    return jsonify(_sanitize_floats(result))


@app.route("/api/journal/trades", methods=["GET"])
def api_journal_trades_list():
    trades = journal.filter_trades(
        symbol=request.args.get("symbol"),
        setup_type=request.args.get("setup_type"),
        status=request.args.get("status"),
        from_date=request.args.get("from_date"),
        to_date=request.args.get("to_date"),
    )
    return jsonify(trades)


@app.route("/api/journal/trades", methods=["POST"])
def api_journal_trades_create():
    body = request.get_json(silent=True) or {}
    try:
        symbol = str(body["symbol"]).strip()
        setup_type = str(body["setup_type"]).strip()
        side = body.get("side", "long")
        entry_price = float(body["entry_price"])
        stop_loss = float(body["stop_loss"])
        target_price = float(body["target_price"])
        quantity = int(body["quantity"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "symbol, setup_type, entry_price, stop_loss, target_price, quantity are required"}), 400
    if not symbol or not setup_type:
        return jsonify({"error": "symbol and setup_type cannot be blank"}), 400

    trade = journal.create_trade(
        symbol, setup_type, side, entry_price, stop_loss, target_price, quantity,
        volume_confirmed=bool(body.get("volume_confirmed", False)),
        higher_tf_aligned=bool(body.get("higher_tf_aligned", False)),
        notes=body.get("notes", ""),
    )
    return jsonify(trade)


@app.route("/api/journal/trades/<trade_id>/close", methods=["PATCH"])
def api_journal_trades_close(trade_id):
    body = request.get_json(silent=True) or {}
    try:
        exit_price = float(body["exit_price"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "exit_price is required"}), 400
    try:
        trade = journal.close_trade(trade_id, exit_price, fees=float(body.get("fees", 0.0)))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(trade)


@app.route("/api/journal/trades/<trade_id>/notes", methods=["PATCH"])
def api_journal_trades_notes(trade_id):
    body = request.get_json(silent=True) or {}
    try:
        trade = journal.update_trade_notes(trade_id, body.get("notes", ""))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(trade)


@app.route("/api/journal/trades/<trade_id>/invalidate", methods=["PATCH"])
def api_journal_trades_invalidate(trade_id):
    try:
        trade = journal.invalidate_trade(trade_id)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(trade)


@app.route("/api/journal/trades/<trade_id>", methods=["DELETE"])
def api_journal_trades_delete(trade_id):
    return jsonify({"trades": journal.delete_trade(trade_id)})


@app.route("/api/journal/analytics", methods=["GET"])
def api_journal_analytics():
    return jsonify(_sanitize_floats(journal.compute_journal_stats()))


@app.route("/api/journal/export", methods=["GET"])
def api_journal_export():
    import csv
    import io

    trades = journal.load_trades()
    buffer = io.StringIO()
    fieldnames = [
        "id", "symbol", "setup_type", "side", "entry_price", "stop_loss", "target_price",
        "quantity", "rr_ratio", "exit_price", "fees", "gross_pl", "net_pl", "status",
        "entry_timestamp", "exit_timestamp", "volume_confirmed", "higher_tf_aligned", "notes",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for t in trades:
        writer.writerow({k: t.get(k) for k in fieldnames})

    from flask import Response
    return Response(
        buffer.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=trade_journal.csv"},
    )


def run(host: str = "127.0.0.1", port: int = 5000) -> None:
    app.run(host=host, port=port)
