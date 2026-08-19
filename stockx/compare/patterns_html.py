import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from stockx.analysis.metrics_snapshot import compute_metrics_snapshot
from stockx.analysis.patterns import (
    MIN_OCCURRENCES_FOR_WIN_RATE,
    PATTERN_REGISTRY,
    PatternStats,
    compute_chart_pattern_stats,
    compute_pattern_stats,
)
from stockx.compare.icons import load_icon
from stockx.compare.layouts import get_layout
from stockx.compare.watchlist import load_watchlist
from stockx.config import REPORTS_DIR, SUPPORTED_INTERVALS
from stockx.strategies.chart_patterns import CHART_PATTERN_NAMES, ChartPatternOccurrence, find_all_chart_patterns

# Confidence-scaled alpha channel: 0.4 (barely-passing match) to 1.0 (textbook match).
DIRECTION_MARKER_STYLE = {
    "bullish": {"position": "belowBar", "shape": "arrowUp", "rgb": "38,166,154"},
    "bearish": {"position": "aboveBar", "shape": "arrowDown", "rgb": "239,83,80"},
    "neutral": {"position": "aboveBar", "shape": "circle", "rgb": "125,130,150"},
}
PATTERN_NAMES = [name for name, _bool_fn, _confidence_fn, _direction in PATTERN_REGISTRY]
ALL_PATTERN_NAMES = PATTERN_NAMES + CHART_PATTERN_NAMES

# 1-3 letter codes shown on the chart's pattern markers (candlestick: a
# "house" pointer at the bar; chart patterns: a label at each occurrence's
# start/end). Bullish/bearish variants of the same candle shape (hammer/
# hanging_man, shooting_star/inverted_hammer, engulfing) intentionally share
# a code -- direction is already conveyed by marker color/position, so this
# is purely "which pattern family is this."
PATTERN_ABBR: Dict[str, str] = {
    "doji": "DOJ", "hammer": "HAM", "hanging_man": "HAM", "shooting_star": "SHS",
    "inverted_hammer": "SHS", "bullish_engulfing": "ENG", "bearish_engulfing": "ENG",
    "morning_star": "MST", "evening_star": "EST", "three_white_soldiers": "3WS", "three_black_crows": "3BC",
    "head_and_shoulders": "H&S", "inverse_head_and_shoulders": "IHS", "double_top": "DT", "double_bottom": "DB",
    "ascending_triangle": "ASC", "descending_triangle": "DSC", "symmetrical_triangle": "SYM",
    "rising_wedge": "RWG", "falling_wedge": "FWG", "flag": "FLG",
}

# Chart patterns get one distinct color per pattern TYPE (not just
# bullish/bearish) so overlapping/adjacent occurrences of different pattern
# types stay visually distinguishable on the chart.
CHART_PATTERN_COLORS: Dict[str, str] = {
    "head_and_shoulders": "#ef5350", "inverse_head_and_shoulders": "#26a69a",
    "double_top": "#ff7043", "double_bottom": "#66bb6a",
    "ascending_triangle": "#42a5f5", "descending_triangle": "#ab47bc",
    "symmetrical_triangle": "#ffca28", "rising_wedge": "#ec407a",
    "falling_wedge": "#26c6da", "flag": "#a1887f",
}

# Icons from lucide-static (MIT), see stockx/compare/icons.py.
ICON_SEARCH = load_icon("search", size=18)
ICON_SHAPES = load_icon("shapes", size=18)
ICON_CHART_HEADER = load_icon("chart-candlestick", size=28)
ICON_TRENDING_UP = load_icon("trending-up", size=14)
ICON_TRENDING_DOWN = load_icon("trending-down", size=14)
ICON_X = load_icon("x", size=12)
_FAVICON_B64 = base64.b64encode(load_icon("chart-candlestick", size=32).encode()).decode()

# Drawing toolbar icons.
ICON_CURSOR = load_icon("mouse-pointer-2", size=18)
ICON_CROSSHAIR = load_icon("crosshair", size=18)
ICON_PEN_LINE = load_icon("pen-line", size=18)
ICON_SLASH = load_icon("slash", size=16)
ICON_MINUS = load_icon("minus", size=16)
ICON_GIT_FORK = load_icon("git-fork", size=16)
ICON_WAVES = load_icon("waves", size=16)
ICON_REPEAT = load_icon("repeat", size=16)
ICON_TRASH = load_icon("trash-2", size=18)

# Reliability flyout panel icons.
ICON_PANEL_RIGHT = load_icon("panel-right", size=18)
ICON_ACTIVITY = load_icon("activity", size=18)

# Watchlist / indicators toolbar icons.
ICON_STAR = load_icon("star", size=16)
ICON_SLIDERS = load_icon("sliders-horizontal", size=18)

# Additional drawing-tool icons.
ICON_RAY = load_icon("arrow-up-right", size=16)
ICON_VLINE = load_icon("separator-vertical", size=16)
ICON_FIB = load_icon("waypoints", size=16)
ICON_RECT = load_icon("rectangle-horizontal", size=16)
ICON_TEXT = load_icon("type", size=16)
ICON_RULER = load_icon("ruler", size=16)

# Backtest / strategy leaderboard icons.
ICON_FLASK = load_icon("flask-conical", size=18)
ICON_TROPHY = load_icon("trophy", size=18)
ICON_RADAR = load_icon("radar", size=16)
ICON_DOWNLOAD = load_icon("download", size=14)

# Trade journal icon.
ICON_NOTEBOOK = load_icon("notebook-pen", size=18)

# Fixed, non-timestamped paths: this is one persistent, growing dashboard
# you keep reopening, not a new artifact every run.
DASHBOARD_JSON_PATH = REPORTS_DIR / "patterns_dashboard_data.json"
DASHBOARD_HTML_PATH = REPORTS_DIR / "patterns_dashboard.html"


def _build_chart_payload(bars: pd.DataFrame) -> dict:
    """Plain OHLCV + pattern-marker data for TradingView's Lightweight
    Charts library (no server-side figure object needed -- it's a pure JS
    renderer). All values cast to native Python types since this goes
    straight through json.dumps."""
    bar_list = []
    volume_list = []
    for ts, row in bars.iterrows():
        t = int(ts.timestamp())
        o, h, l, c, v = (float(row["open"]), float(row["high"]), float(row["low"]),
                         float(row["close"]), float(row["volume"]))
        bar_list.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume_list.append({"time": t, "value": v, "color": "#26a69a" if c >= o else "#ef5350"})

    patterns: Dict[str, list] = {}
    for name, bool_fn, confidence_fn, direction in PATTERN_REGISTRY:
        occurred = bool_fn(bars)
        if not occurred.any():
            continue
        occ_bars = bars[occurred]
        confidence = confidence_fn(bars)[occurred]
        style = DIRECTION_MARKER_STYLE[direction]

        markers = []
        for ts, conf in zip(occ_bars.index, confidence.values):
            alpha = 0.4 + float(conf) / 100 * 0.6
            markers.append({
                "time": int(ts.timestamp()),
                "position": style["position"],
                "shape": style["shape"],
                "color": f"rgba({style['rgb']},{alpha:.2f})",
            })
        patterns[name] = markers

    return {"bars": bar_list, "volume": volume_list, "patterns": patterns}


def _build_chart_pattern_payload(all_chart_patterns: Dict[str, List[ChartPatternOccurrence]]) -> dict:
    """Chart patterns can't be represented as point markers alone -- each
    occurrence carries its defining vertices and trendline segments so the
    dashboard can draw the actual shape (neckline, support/resistance
    lines), not just a single confirmation marker."""
    payload: Dict[str, list] = {}
    for name, occurrences in all_chart_patterns.items():
        if not occurrences:
            continue
        items = []
        for occ in occurrences:
            items.append({
                "direction": occ.direction,
                "vertices": [{"time": int(t.timestamp()), "value": float(p)} for t, p in occ.vertices],
                "trendlines": [
                    [{"time": int(a[0].timestamp()), "value": float(a[1])},
                     {"time": int(b[0].timestamp()), "value": float(b[1])}]
                    for a, b in occ.trendlines
                ],
                "breakout": {"time": int(occ.breakout_time.timestamp()), "value": float(occ.breakout_price)},
                "confidence": float(occ.confidence),
            })
        payload[name] = items
    return payload


# --- Pattern glyphs ---------------------------------------------------------
# Small schematic SVG sketches (viewBox 0 0 40 28) standing in for the pattern
# name in the reliability panel. Candlestick glyphs use the same historically
# conventional per-candle coloring you'd see in any candlestick cheat sheet
# (fixed per pattern, not tied to the row's direction, since e.g. bullish/
# bearish engulfing need two different-colored candles in the same glyph).
# Chart patterns are geometric, not candle-colored, so they're sketched as
# plain accent-blue line drawings (same blue as the drawing toolbar's trend
# lines) to visually read as "structural" rather than "candle" patterns --
# reinforced by the Type column icon sitting right next to them.
_GREEN = "#26a69a"
_RED = "#ef5350"
_GRAY = "#7d8296"
_BLUE = "#2962ff"


def _candle(x: float, body_top: float, body_bottom: float, wick_top: float, wick_bottom: float, color: str, w: float = 7) -> str:
    body_h = max(body_bottom - body_top, 1.5)
    return (
        f'<line x1="{x}" y1="{wick_top}" x2="{x}" y2="{wick_bottom}" stroke="{color}" stroke-width="1.3"/>'
        f'<rect x="{x - w / 2}" y="{body_top}" width="{w}" height="{body_h}" fill="{color}"/>'
    )


def _sketch_lines(*segments: List[Tuple[float, float]]) -> str:
    parts = []
    for points in segments:
        pts = " ".join(f"{x},{y}" for x, y in points)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{_BLUE}" stroke-width="2" '
                      f'stroke-linecap="round" stroke-linejoin="round"/>')
    return "".join(parts)


def _glyph(inner: str) -> str:
    return f'<svg width="40" height="28" viewBox="0 0 40 28" xmlns="http://www.w3.org/2000/svg">{inner}</svg>'


PATTERN_ICON_SVG: Dict[str, str] = {
    "doji": _glyph(_candle(20, 13, 15, 5, 23, _GRAY)),
    "hammer": _glyph(_candle(20, 7, 12, 6, 24, _GREEN)),
    "hanging_man": _glyph(_candle(20, 7, 12, 6, 24, _RED)),
    "shooting_star": _glyph(_candle(20, 17, 22, 4, 23, _RED)),
    "inverted_hammer": _glyph(_candle(20, 17, 22, 4, 23, _GREEN)),
    "bullish_engulfing": _glyph(_candle(13, 13, 17, 11, 19, _RED) + _candle(27, 6, 23, 4, 25, _GREEN)),
    "bearish_engulfing": _glyph(_candle(13, 13, 17, 11, 19, _GREEN) + _candle(27, 6, 23, 4, 25, _RED)),
    "morning_star": _glyph(
        _candle(8, 6, 19, 4, 21, _RED) + _candle(20, 18, 20, 16, 22, _GRAY) + _candle(32, 7, 20, 5, 22, _GREEN)
    ),
    "evening_star": _glyph(
        _candle(8, 7, 20, 5, 22, _GREEN) + _candle(20, 6, 8, 4, 10, _GRAY) + _candle(32, 6, 19, 4, 21, _RED)
    ),
    "three_white_soldiers": _glyph(
        _candle(8, 17, 23, 15, 25, _GREEN) + _candle(20, 11, 17, 9, 19, _GREEN) + _candle(32, 5, 11, 3, 13, _GREEN)
    ),
    "three_black_crows": _glyph(
        _candle(8, 5, 11, 3, 13, _RED) + _candle(20, 11, 17, 9, 19, _RED) + _candle(32, 17, 23, 15, 25, _RED)
    ),
    "head_and_shoulders": _glyph(_sketch_lines([(3, 20), (9, 8), (15, 16), (20, 4), (25, 16), (31, 8), (37, 20)])),
    "inverse_head_and_shoulders": _glyph(_sketch_lines([(3, 8), (9, 20), (15, 12), (20, 24), (25, 12), (31, 20), (37, 8)])),
    "double_top": _glyph(_sketch_lines([(3, 20), (12, 6), (20, 16), (28, 6), (37, 20)])),
    "double_bottom": _glyph(_sketch_lines([(3, 8), (12, 22), (20, 12), (28, 22), (37, 8)])),
    "ascending_triangle": _glyph(_sketch_lines([(4, 6), (36, 6)], [(4, 24), (36, 8)])),
    "descending_triangle": _glyph(_sketch_lines([(4, 22), (36, 22)], [(4, 4), (36, 20)])),
    "symmetrical_triangle": _glyph(_sketch_lines([(4, 4), (36, 14)], [(4, 24), (36, 14)])),
    "rising_wedge": _glyph(_sketch_lines([(4, 23), (36, 6)], [(4, 26), (36, 11)])),
    "falling_wedge": _glyph(_sketch_lines([(4, 4), (36, 17)], [(4, 11), (36, 20)])),
    "flag": _glyph(_sketch_lines([(4, 25), (13, 5)], [(13, 5), (36, 10)], [(13, 10), (36, 15)])),
}

ICON_TYPE_CANDLESTICK = load_icon("chart-candlestick", size=18)
ICON_TYPE_CHART = load_icon("shapes", size=18)


_DIRECTION_SECTION_TITLES = {
    "bullish": "Bullish patterns",
    "bearish": "Bearish patterns",
    "neutral": "Neutral patterns",
}


def _stat_cell(bar_html: str, label: str) -> str:
    return f'<div class="stat-cell"><div class="stat-bar-track">{bar_html}</div><div class="stat-cell-label">{label}</div></div>'


def _pattern_cell(name: str) -> str:
    icon = PATTERN_ICON_SVG.get(name, "")
    return (
        f'<div class="pattern-glyph-cell" data-pattern-name="{name}" '
        f'style="display:flex; flex-direction:column; align-items:center; gap:2px; max-width:150px; margin:0 auto;" '
        f'title="Click to toggle {name} on the chart">'
        f'{icon}<div style="font-size:10px; color:#7d8296; text-align:center; word-break:break-word; line-height:1.25;">{name}</div>'
        f"</div>"
    )


def _type_cell(kind: str) -> str:
    if kind == "Candlestick":
        return f'<div style="display:flex; justify-content:center; color:#7d8296;" title="Candlestick pattern">{ICON_TYPE_CANDLESTICK}</div>'
    return f'<div style="display:flex; justify-content:center; color:#2962ff;" title="Chart pattern">{ICON_TYPE_CHART}</div>'


def _occurrences_cell(occurrences: int, max_occ: int) -> str:
    pct = 0 if max_occ == 0 else occurrences / max_occ * 100
    bar = f'<div class="stat-bar-fill" style="width:{pct:.0f}%; background:#5c6bc0;"></div>'
    return _stat_cell(bar, str(occurrences))


def _win_rate_cell(s: PatternStats) -> str:
    if s.occurrences == 0:
        return _stat_cell("", "N/A")
    if pd.isna(s.win_rate):
        return _stat_cell("", "neutral")
    pct = s.win_rate * 100
    color = "#26a69a" if s.win_rate >= 0.5 else "#ef5350"
    low_sample = s.occurrences < MIN_OCCURRENCES_FOR_WIN_RATE
    opacity = 0.4 if low_sample else 1.0
    bar = f'<div class="stat-bar-fill" style="width:{pct:.0f}%; background:{color}; opacity:{opacity};"></div>'
    label = f"{pct:.0f}%" + (f" (n={s.occurrences})" if low_sample else "")
    return _stat_cell(bar, label)


def _return_cell(value: float, max_abs: float) -> str:
    if pd.isna(value):
        return _stat_cell('<div class="stat-bar-center"></div>', "N/A")
    frac = 0.0 if max_abs == 0 else max(-1.0, min(1.0, value / max_abs))
    pct = abs(frac) * 50
    color = "#26a69a" if value >= 0 else "#ef5350"
    side = f"left:50%; width:{pct:.0f}%;" if value >= 0 else f"right:50%; width:{pct:.0f}%;"
    bar = f'<div class="stat-bar-center"></div><div class="stat-bar-fill" style="{side} background:{color};"></div>'
    return _stat_cell(bar, f"{value:+.2%}")


def _confidence_cell(value: float) -> str:
    if pd.isna(value):
        return _stat_cell("", "N/A")
    bar = f'<div class="stat-bar-fill" style="width:{value:.0f}%; background:#42a5f5;"></div>'
    return _stat_cell(bar, f"{value:.0f}/100")


def _stats_row_html(s: PatternStats, kind: str, max_occ: int, max_abs_return: float) -> str:
    return (
        f"<tr>"
        f"<td>{_pattern_cell(s.name)}</td>"
        f"<td>{_type_cell(kind)}</td>"
        f"<td>{_occurrences_cell(s.occurrences, max_occ)}</td>"
        f"<td>{_win_rate_cell(s)}</td>"
        f"<td>{_return_cell(s.avg_forward_return, max_abs_return)}</td>"
        f"<td>{_confidence_cell(s.avg_confidence)}</td>"
        f"</tr>"
    )


def _stats_table_html(
    pattern_stats: List[PatternStats], chart_pattern_stats: List[PatternStats], forward_bars: int,
) -> str:
    """Reliability stats grouped by direction (bullish/bearish/neutral)
    rather than by pattern type -- candlestick and chart patterns of the
    same direction sit in one section, sorted by win rate descending (most
    reliable first), so the panel reads as "what's the market telling me"
    rather than "candlestick stats, then a second unrelated chart-pattern
    table". Every column but Pattern/Type is rendered as a small bar/gauge
    rather than raw text -- Occurrences and Avg confidence scale 0..max (or
    0..100), Win rate is a green/red fill, Avg forward return is a diverging
    bar around a center zero-line -- with the exact number kept as a small
    caption underneath each bar for precision."""
    tagged = [(s, "Candlestick") for s in pattern_stats] + [(s, "Chart") for s in chart_pattern_stats]
    max_occ = max((s.occurrences for s, _ in tagged), default=1) or 1
    max_abs_return = max((abs(s.avg_forward_return) for s, _ in tagged if not pd.isna(s.avg_forward_return)), default=1.0) or 1.0

    buckets: Dict[str, list] = {"bullish": [], "bearish": [], "neutral": []}
    for s, kind in tagged:
        buckets.setdefault(s.direction, []).append((s, kind))

    sections = []
    for key in ("bullish", "bearish", "neutral"):
        rows_data = buckets.get(key, [])
        if not rows_data:
            continue
        rows_data.sort(key=lambda item: -1 if pd.isna(item[0].win_rate) else item[0].win_rate, reverse=True)
        rows = "".join(_stats_row_html(s, kind, max_occ, max_abs_return) for s, kind in rows_data)
        sections.append(f"""
        <h3>{_DIRECTION_SECTION_TITLES[key]}</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%;">
          <tr><th>Pattern</th><th>Type</th><th>Occurrences</th><th>Win rate</th>
              <th>Avg forward return</th><th>Avg confidence</th></tr>
          {rows}
        </table>
        """)

    return f"""
    <h2>Historical pattern reliability (empirical, this symbol only)</h2>
    <p>Win rate = fraction of past occurrences followed by a move in the pattern's
    stated direction over the next {forward_bars} bars. Not a statistical
    guarantee &mdash; treat rows flagged &ldquo;insufficient data&rdquo; as noise, not signal.
    Sections sorted by win rate, most reliable first.</p>
    {''.join(sections)}
    """


def _load_dashboard_data() -> Dict[str, dict]:
    if not DASHBOARD_JSON_PATH.exists():
        return {}
    with open(DASHBOARD_JSON_PATH) as f:
        return json.load(f)


def _save_dashboard_data(data: Dict[str, dict]) -> None:
    DASHBOARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_JSON_PATH, "w") as f:
        json.dump(data, f)


def _render_dashboard_html(data: Dict[str, dict], watchlist: List[str] = None) -> str:
    watchlist = watchlist or []

    # Entries are keyed "SYMBOL@interval" so one symbol can have several
    # timeframes cached side by side; the datalist/search only ever deals
    # in plain symbol names, deduplicated across whatever intervals exist.
    symbols = sorted({entry["symbol"] for entry in data.values()})
    default_key = max(data, key=lambda k: data[k]["generated_at"]) if data else None

    symbol_datalist_options = "".join(f'<option value="{s}">' for s in symbols)
    pattern_datalist_options = "".join(f'<option value="{p}">' for p in ALL_PATTERN_NAMES)
    interval_buttons = "".join(
        f'<button class="tool-btn interval-btn{" tool-btn-active" if i == "1h" else ""}" '
        f'data-interval="{i}" style="width:auto; height:28px; padding:0 12px; font-size:13px;">{i}</button>'
        for i in SUPPORTED_INTERVALS
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Candlestick pattern dashboard</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{_FAVICON_B64}">
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {{
      --bg: #0f1117; --bg-panel: #1a1d27; --bg-elevated: #262a37; --bg-hover: #2f3444;
      --border: #2f3444; --text: #d1d4dc; --text-dim: #7d8296;
      --accent: #2962ff; --green: #26a69a; --red: #ef5350;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg); color: var(--text); margin:0; padding:10px 12px 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    #chart-widget {{ border:1px solid var(--border); border-radius:8px; background:#0f1117; overflow:hidden; }}
    #chart-header-bar {{
      display:flex; align-items:center; flex-wrap:wrap; gap:16px; padding:8px 12px;
      background:var(--bg-panel); border-bottom:1px solid var(--border);
    }}
    #chart-status-bar {{ display:flex; align-items:center; flex-wrap:wrap; gap:12px; padding:4px 12px; }}
    h1, h2, h3 {{ color: var(--text); font-weight: 600; }}
    p {{ color: var(--text-dim); line-height:1.5; }}
    code {{ background: var(--bg-elevated); color: var(--text); padding:2px 6px; border-radius:4px; font-size:0.9em; }}
    input {{
      background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; outline: none;
    }}
    input::placeholder {{ color: var(--text-dim); }}
    input:focus {{ border-color: var(--accent); }}
    button {{
      background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; cursor: pointer; font-family: inherit;
    }}
    button:hover {{ background: var(--bg-hover); }}
    table {{ border-collapse: collapse; }}
    table, th, td {{ border: 1px solid var(--border); }}
    th, td {{ padding: 6px 10px; text-align:left; }}
    th {{ background: var(--bg-elevated); color: var(--text); }}
    tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--bg-elevated); border-radius: 5px; }}

    #drawing-toolbar button.tool-btn {{
      width: 34px; height: 34px; display:flex; align-items:center; justify-content:center;
      background: transparent; border: 1px solid transparent; border-radius: 4px; color:var(--text); padding:0;
    }}
    #drawing-toolbar button.tool-btn:hover {{ background: var(--bg-hover); }}
    #drawing-toolbar button.tool-btn.tool-btn-active {{ background: rgba(41,98,255,0.15); border-color: var(--accent); color:var(--accent); }}
    #drawing-toolbar .toolbar-divider {{ border-top: 1px solid var(--border); margin: 4px 2px; }}
    .tool-flyout {{
      display:none; position:absolute; left:40px; top:0; background:var(--bg-panel); border:1px solid var(--border); border-radius:6px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4); padding:4px; z-index:10; min-width:180px;
      max-height:90vh; overflow-y:auto;
    }}
    .tool-flyout button.flyout-item {{
      display:flex; align-items:center; gap:8px; width:100%; padding:6px 10px; background:none; border:none;
      text-align:left; font-size:13px; color:var(--text); box-sizing:border-box;
    }}
    .tool-flyout button.flyout-item:hover:not(:disabled) {{ background:var(--bg-hover); }}
    .tool-flyout button.flyout-item:disabled {{ color:var(--text-dim); cursor:not-allowed; }}
    .tool-flyout .flyout-indicator-form {{ padding:6px 8px; border-top:1px solid var(--border); }}
    .tool-flyout .flyout-indicator-form:first-child {{ border-top:none; }}
    .tool-flyout .flyout-indicator-form label {{
      display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:12px;
      color:var(--text-dim); margin-bottom:4px;
    }}
    .tool-flyout .flyout-indicator-form input {{ width:52px; padding:2px 4px; font-size:12px; }}
    .tool-flyout .flyout-indicator-form button {{
      width:100%; margin-top:2px; padding:4px 0; font-size:12px; background:rgba(41,98,255,0.12);
      border-color:var(--accent); color:var(--accent);
    }}
    .tool-flyout .flyout-indicator-form button:hover {{ background:rgba(41,98,255,0.22); }}

    #watchlist-panel.side-panel {{ width:320px; }}
    #watchlist-star-btn {{ display:flex; align-items:center; justify-content:center; width:26px; height:26px; color:var(--text-dim); }}
    #watchlist-star-btn.watchlist-star-active {{ color:#ffca28; border-color:#ffca28; }}
    #watchlist-star-btn.watchlist-star-active svg {{ fill:currentColor; }}
    #watchlist-list .watchlist-empty-hint {{ padding:10px; font-size:12px; color:var(--text-dim); }}
    .watchlist-item {{
      display:flex; align-items:center; justify-content:space-between; padding:7px 10px; cursor:pointer;
      font-size:13px; border-bottom:1px solid var(--border); border-radius:4px;
    }}
    .watchlist-item:hover {{ background:var(--bg-hover); }}
    .watchlist-item.watchlist-item-active {{ background:rgba(41,98,255,0.12); }}
    .watchlist-item .watchlist-remove-btn {{
      display:flex; align-items:center; justify-content:center; width:18px; height:18px;
      background:none; border:none; color:var(--text-dim); padding:0;
    }}
    .watchlist-item .watchlist-remove-btn:hover {{ color:var(--red); background:none; }}

    #scan-results {{ margin-bottom:10px; }}
    #scan-results .scan-hint {{ font-size:12px; color:var(--text-dim); padding:4px 2px; }}
    .scan-hit {{
      display:flex; align-items:center; justify-content:space-between; gap:8px; padding:7px 10px;
      cursor:pointer; font-size:12px; border:1px solid var(--border); border-radius:6px; margin-bottom:4px;
    }}
    .scan-hit:hover {{ background:var(--bg-hover); }}
    .scan-hit .scan-hit-left {{ display:flex; flex-direction:column; gap:1px; min-width:0; }}
    .scan-hit .scan-hit-symbol {{ font-weight:700; color:var(--text); }}
    .scan-hit .scan-hit-pattern {{ color:var(--text-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .scan-hit .scan-hit-winrate {{ font-weight:700; white-space:nowrap; }}

    .journal-form label {{
      display:flex; flex-direction:column; gap:2px; font-size:11px; color:var(--text-dim); margin-bottom:6px;
    }}
    .journal-form label input, .journal-form label select {{ padding:4px 6px; font-size:12px; }}
    .journal-form .journal-form-row {{ display:flex; gap:8px; }}
    .journal-form .journal-form-row label {{ flex:1; min-width:0; }}
    .journal-form label.journal-checkbox {{ flex-direction:row; align-items:center; gap:6px; font-size:12px; color:var(--text); }}
    #journal-validation {{ font-size:12px; padding:8px; border-radius:6px; margin:8px 0; border:1px solid var(--border); }}
    #journal-validation .journal-error {{ color:var(--red); }}
    #journal-validation .journal-warning {{ color:#ffca28; }}
    #journal-validation.journal-valid {{ border-color:var(--green); }}
    #journal-validation.journal-invalid {{ border-color:var(--red); }}
    .journal-tiles {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; }}
    .journal-tile {{
      flex:1; min-width:90px; padding:8px 10px; border:1px solid var(--border); border-radius:6px; background:#12141c;
    }}
    .journal-tile .journal-tile-label {{ font-size:10px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.4px; }}
    .journal-tile .journal-tile-value {{ font-size:16px; font-weight:700; margin-top:2px; }}
    .journal-trade-row td {{ padding:5px 6px; font-size:12px; border-bottom:1px solid var(--border); }}
    .journal-trade-row button {{ font-size:10px; padding:2px 6px; margin-right:4px; }}
    #journal-filters {{ display:flex; gap:6px; margin:10px 0; flex-wrap:wrap; }}
    #journal-filters select, #journal-filters input {{ font-size:11px; padding:3px 5px; }}

    .backtest-trade-row:hover {{ background:var(--bg-hover); }}
    #backtest-trades-panel table th {{ position:sticky; top:0; z-index:1; }}

    .pane-toggle-btn.tool-btn-active {{ background: rgba(41,98,255,0.15); border-color: var(--accent); color:var(--accent); }}
    .oscillator-pane {{ border-top:1px solid #2f3444; margin-top:4px; }}
    .oscillator-pane-label {{
      font-size:11px; color:#7d8296; padding:2px 6px; font-family:monospace;
    }}

    .side-tab {{
      position: fixed; right: 0; z-index: 101; display:flex; align-items:center; gap:6px;
      background:var(--bg-panel); border:1px solid var(--border); border-right:none; border-radius: 6px 0 0 6px;
      padding: 8px 10px; cursor:pointer; color:var(--text); box-shadow: -2px 0 8px rgba(0,0,0,0.3);
      opacity: 1; transition: opacity 0.15s ease;
    }}
    .side-tab:hover {{ background:var(--bg-hover); }}
    /* While a panel is open, the tabs (fixed on the right edge, above the
       panel's z-index) sit directly over the panel's own content -- dim
       them so whatever's underneath (e.g. watchlist rows) stays readable,
       restoring full opacity on hover so they're still easy to aim for. */
    .side-tab.side-tab-dimmed {{ opacity: 0.28; }}
    .side-tab.side-tab-dimmed:hover {{ opacity: 1; }}
    #watchlist-tab {{ top: 94px; }}
    #reliability-tab {{ top: 140px; }}
    #oscillators-tab {{ top: 186px; }}
    #strategies-tab {{ top: 232px; }}
    #journal-tab {{ top: 278px; }}
    #panel-backdrop {{
      display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:99;
    }}
    #panel-backdrop.backdrop-visible {{ display:block; }}
    .side-panel {{
      position: fixed; top:0; right:0; height:100vh; width:680px; max-width:94vw;
      background:var(--bg-panel); border-left:1px solid var(--border); box-shadow:-4px 0 20px rgba(0,0,0,0.5);
      overflow-y:auto; overflow-x:hidden; padding:16px 20px; z-index:100; transform:translateX(100%);
      transition: transform 0.25s ease; box-sizing:border-box;
    }}
    .side-panel.panel-open {{ transform:translateX(0); }}
    #reliability-panel h2 {{ font-size:15px; margin:6px 0 8px; }}
    #reliability-panel h3 {{
      font-size:12px; margin:18px 0 8px; text-transform:uppercase; letter-spacing:0.5px; color:#9aa1b4;
    }}
    #reliability-panel p {{ font-size:12px; margin:0 0 8px; }}
    #reliability-panel table {{ font-size:12px; table-layout:fixed; width:100%; }}
    #reliability-panel th {{ text-align:center; font-size:10.5px; padding:6px 3px; }}
    #reliability-panel td {{ vertical-align:middle; text-align:center; padding:6px 3px; overflow:hidden; }}
    #reliability-panel th:nth-child(1), #reliability-panel td:nth-child(1) {{ width:26%; }}
    #reliability-panel th:nth-child(2), #reliability-panel td:nth-child(2) {{ width:9%; }}
    #reliability-panel th:nth-child(3), #reliability-panel td:nth-child(3) {{ width:16%; }}
    #reliability-panel th:nth-child(4), #reliability-panel td:nth-child(4) {{ width:16%; }}
    #reliability-panel th:nth-child(5), #reliability-panel td:nth-child(5) {{ width:17%; }}
    #reliability-panel th:nth-child(6), #reliability-panel td:nth-child(6) {{ width:16%; }}
    .stat-cell {{ display:flex; flex-direction:column; align-items:center; gap:3px; }}
    .stat-bar-track {{
      position:relative; width:100%; min-width:56px; height:7px; border-radius:4px;
      background:rgba(255,255,255,0.08); overflow:hidden;
    }}
    .stat-bar-fill {{ position:absolute; top:0; height:100%; border-radius:4px; }}
    .stat-bar-center {{ position:absolute; top:0; left:50%; width:1px; height:100%; background:rgba(255,255,255,0.25); }}
    .stat-cell-label {{ font-size:10px; color:#7d8296; white-space:nowrap; }}
    .pattern-glyph-cell {{
      cursor:pointer; border-radius:6px; padding:4px; border:1px solid transparent;
      transition: background 0.15s, border-color 0.15s;
    }}
    .pattern-glyph-cell:hover {{ background: var(--bg-hover); }}
    .pattern-glyph-cell.pattern-active {{ background: rgba(41,98,255,0.18); border-color: var(--accent); }}

    #oscillators-panel {{ height:auto; max-height:92vh; display:flex; flex-direction:column; }}
    .osc-grid {{
      display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: auto auto auto;
      grid-template-areas: "tl top tr" "ml center mr" "bl bottom br";
      gap: 14px; justify-items: center; align-items: center;
      padding: 8px 0;
    }}
    .osc-card {{
      background:#1a1d27; border:1px solid #2f3444; border-radius:8px; padding:8px 10px 10px;
      min-width:140px; text-align:center; transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }}
    .osc-card:hover {{ transform: translateY(-2px); border-color:#454b5c; background:#1d212c; }}
    .osc-card.osc-hero {{
      background:#1c2233; border:1px solid var(--accent); padding:14px 16px 16px; min-width:180px;
      box-shadow: 0 0 24px rgba(41,98,255,0.15);
    }}
    .osc-card.osc-hero:hover {{ background:#20273a; box-shadow: 0 0 32px rgba(41,98,255,0.25); }}

    #marker-overlay {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; overflow:hidden; }}
    .house-marker {{
      position:absolute; top:0; left:0; min-width:18px; font-size:9px; font-weight:700;
      color:#0f1117; text-align:center; line-height:1.1; box-sizing:border-box; white-space:nowrap; padding:0 3px;
    }}
    .house-up {{ clip-path: polygon(50% 0%, 100% 38%, 100% 100%, 0% 100%, 0% 38%); padding-top:7px; padding-bottom:2px; }}
    .house-down {{ clip-path: polygon(0% 0%, 100% 0%, 100% 62%, 50% 100%, 0% 62%); padding-bottom:7px; padding-top:2px; }}
    .chart-pattern-label {{
      position:absolute; top:0; left:0; padding:1px 5px; font-size:9px; font-weight:700;
      background:rgba(15,17,23,0.85); border:1px solid; border-radius:3px; white-space:nowrap;
    }}
    #text-annotation-overlay {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; overflow:hidden; }}
    .text-annotation {{
      position:absolute; top:0; left:0; padding:2px 6px; font-size:11px; font-weight:600;
      background:rgba(41,98,255,0.15); color:#8ab4ff; border:1px solid #2962ff; border-radius:4px; white-space:nowrap;
    }}
    .text-annotation.measurement-label {{ background:rgba(255,202,40,0.15); color:#ffca28; border-color:#ffca28; font-weight:500; }}
  </style>
</head>
<body>
  <div id="chart-widget">
    <div id="chart-header-bar">
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="color:var(--accent); display:flex;">{ICON_CHART_HEADER}</span>
        <span style="color:#7d8296; display:flex;">{ICON_SEARCH}</span>
        <input id="symbol-search-box" list="symbol-search-options" placeholder="Search symbol..."
               style="font-size:14px; padding:5px 8px; width:180px;" autocomplete="off">
        <button id="symbol-search-button" style="font-size:14px; padding:5px 10px;">Search</button>
      </div>
      <div id="interval-toolbar" style="display:flex; gap:4px;">{interval_buttons}</div>
      <div style="display:flex; align-items:center; gap:8px;">
        <div id="current-symbol-label" style="font-weight:bold; font-size:13px;"></div>
        <button id="watchlist-star-btn" title="Add/remove current symbol from watchlist" disabled>{ICON_STAR}</button>
      </div>
    </div>
    <datalist id="symbol-search-options">{symbol_datalist_options}</datalist>
    <div id="symbol-search-hint" style="color:#7d8296; min-height:18px; font-size:12px; padding:2px 12px 0;"></div>

    <div id="chart-status-bar">
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="color:#7d8296; display:flex;">{ICON_SHAPES}</span>
        <input id="pattern-search-box" list="pattern-search-options" placeholder="Toggle pattern..."
               style="font-size:13px; padding:4px 8px; width:160px;" autocomplete="off">
        <button id="pattern-search-button" style="font-size:13px; padding:4px 8px;">Toggle</button>
      </div>
      <span style="font-size:12px; color:#7d8296;">Panes:</span>
      <button class="tool-btn pane-toggle-btn" data-pane="RSI" style="width:auto; height:26px; padding:0 10px; font-size:12px;">RSI</button>
      <button class="tool-btn pane-toggle-btn" data-pane="MACD" style="width:auto; height:26px; padding:0 10px; font-size:12px;">MACD</button>
      <button class="tool-btn pane-toggle-btn" data-pane="Stoch" style="width:auto; height:26px; padding:0 10px; font-size:12px;">Stoch</button>
      <div id="ohlc-legend" style="font-family:monospace; font-size:12px; min-height:16px;"></div>
      <div id="tool-hint" style="font-size:12px; color:var(--accent); min-height:16px;"></div>
    </div>
    <datalist id="pattern-search-options">{pattern_datalist_options}</datalist>
    <div id="pattern-search-hint" style="color:#7d8296; min-height:16px; font-size:12px; padding:0 12px;"></div>
    <div id="active-patterns" style="padding:0 12px;"></div>
    <div id="active-indicators" style="padding:0 12px;"></div>

    <div style="display:flex; align-items:stretch; gap:0;">
      <div id="drawing-toolbar" style="display:flex; flex-direction:column; gap:2px; padding:6px;
           border-right:1px solid var(--border); background:var(--bg-panel);">
        <button class="tool-btn tool-btn-active" data-tool="cursor" title="Cursor">{ICON_CURSOR}</button>
        <button class="tool-btn" data-tool="crosshair" title="Cross cursor">{ICON_CROSSHAIR}</button>
        <div class="toolbar-divider"></div>
        <div style="position:relative;">
          <button class="tool-btn" id="lines-group-trigger" title="Lines / Channels / Pitchforks">{ICON_PEN_LINE}</button>
          <div id="lines-panel" class="tool-flyout">
            <button class="flyout-item" data-tool="trendline">{ICON_SLASH} Trend Line</button>
            <button class="flyout-item" data-tool="ray">{ICON_RAY} Ray</button>
            <button class="flyout-item" data-tool="hline">{ICON_MINUS} Horizontal Line</button>
            <button class="flyout-item" data-tool="vline">{ICON_VLINE} Vertical Line</button>
            <button class="flyout-item" data-tool="channel">{ICON_PEN_LINE} Channel</button>
            <button class="flyout-item" data-tool="pitchfork">{ICON_GIT_FORK} Pitchfork</button>
            <div class="toolbar-divider"></div>
            <button class="flyout-item" data-tool="rect">{ICON_RECT} Rectangle</button>
            <div class="toolbar-divider"></div>
            <button class="flyout-item" data-tool="fib">{ICON_FIB} Fib Retracement</button>
            <button class="flyout-item" data-tool="fibext">{ICON_FIB} Fib Extension</button>
            <button class="flyout-item" data-tool="fibfan">{ICON_FIB} Fib Fan</button>
            <div class="toolbar-divider"></div>
            <button class="flyout-item" data-tool="text">{ICON_TEXT} Text Note</button>
            <button class="flyout-item" data-tool="measure">{ICON_RULER} Measure</button>
          </div>
        </div>
        <div style="position:relative;">
          <button class="tool-btn" id="patterns-group-trigger" title="Chart Patterns / Elliott Wave / Cycles">{ICON_SHAPES}</button>
          <div id="patterns-panel" class="tool-flyout">
            <button class="flyout-item" disabled title="Coming soon">{ICON_SHAPES} Chart Pattern (soon)</button>
            <button class="flyout-item" disabled title="Coming soon">{ICON_WAVES} Elliott Wave (soon)</button>
            <button class="flyout-item" disabled title="Coming soon">{ICON_REPEAT} Cycles (soon)</button>
          </div>
        </div>
        <div style="position:relative;">
          <button class="tool-btn" id="indicators-group-trigger" title="Indicators">{ICON_SLIDERS}</button>
          <div id="indicators-panel" class="tool-flyout" style="min-width:170px;">
            <div class="flyout-indicator-form">
              <label>SMA period <input type="number" id="ind-sma-period" value="20" min="2" max="500"></label>
              <button data-indicator="SMA">{ICON_SLIDERS} Add SMA</button>
            </div>
            <div class="flyout-indicator-form">
              <label>EMA period <input type="number" id="ind-ema-period" value="50" min="2" max="500"></label>
              <button data-indicator="EMA">{ICON_SLIDERS} Add EMA</button>
            </div>
            <div class="flyout-indicator-form">
              <label>BB period <input type="number" id="ind-bb-period" value="20" min="2" max="500"></label>
              <label>BB stddev <input type="number" id="ind-bb-stddev" value="2" min="0.5" max="5" step="0.5"></label>
              <button data-indicator="BB">{ICON_SLIDERS} Add Bollinger Bands</button>
            </div>
            <div class="flyout-indicator-form">
              <button data-indicator="VWAP">{ICON_SLIDERS} Add VWAP</button>
            </div>
          </div>
        </div>
        <div style="position:relative;">
          <button class="tool-btn" id="backtest-group-trigger" title="Backtest">{ICON_FLASK}</button>
          <div id="backtest-panel" class="tool-flyout" style="min-width:200px;">
            <div class="flyout-indicator-form">
              <label>Strategy
                <select id="backtest-strategy-select" style="width:100%; margin-top:4px; padding:3px 4px; font-size:12px;"></select>
              </label>
              <label title="When a signal's order is considered filled: at the next bar's open (no lookahead, realistic) or at the same bar's own close (optimistic, faster iteration).">Execution timing
                <select id="backtest-execution-timing-select" style="width:100%; margin-top:4px; padding:3px 4px; font-size:12px;">
                  <option value="next_open">Next bar open</option>
                  <option value="same_close">Same bar close</option>
                </select>
              </label>
              <label title="Only affects strategies with a stop-loss/take-profit level: which one fires first when both fall inside the same bar's high-low range, since OHLC data alone doesn't record tick order.">Bar path (stop/target order)
                <select id="backtest-intrabar-path-select" style="width:100%; margin-top:4px; padding:3px 4px; font-size:12px;">
                  <option value="ohlc">Open → High → Low → Close</option>
                  <option value="olhc">Open → Low → High → Close</option>
                </select>
              </label>
              <div id="backtest-params-form"></div>
              <button id="backtest-run-btn">{ICON_FLASK} Run Backtest</button>
            </div>
            <div class="flyout-indicator-form">
              <button id="backtest-compare-btn">{ICON_TROPHY} Compare All Strategies</button>
            </div>
          </div>
        </div>
        <div class="toolbar-divider"></div>
        <button class="tool-btn" id="clear-drawings-btn" title="Clear all drawings">{ICON_TRASH}</button>
      </div>
      <div style="flex:1; min-width:0; padding:4px;">
        <div id="chart" style="width:100%;"></div>
        <div id="oscillator-panes"></div>
        <div id="backtest-results"></div>
      </div>
    </div>
  </div>

  <div id="recommendation" style="margin:10px 0; padding:12px 14px; border-radius:8px; background:#1a1d27; border:1px solid #2f3444;"></div>

  <div id="watchlist-tab" class="side-tab" title="Watchlist">{ICON_STAR}<span style="font-size:13px;">Watchlist</span></div>
  <div id="reliability-tab" class="side-tab" title="Pattern reliability">{ICON_PANEL_RIGHT}<span style="font-size:13px;">Reliability</span></div>
  <div id="oscillators-tab" class="side-tab" title="Oscillators &amp; metrics">{ICON_ACTIVITY}<span style="font-size:13px;">Oscillators</span></div>
  <div id="strategies-tab" class="side-tab" title="Strategy leaderboard">{ICON_TROPHY}<span style="font-size:13px;">Strategies</span></div>
  <div id="journal-tab" class="side-tab" title="Trade journal">{ICON_NOTEBOOK}<span style="font-size:13px;">Journal</span></div>
  <div id="panel-backdrop"></div>

  <div id="watchlist-panel" class="side-panel">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <strong>Watchlist</strong>
      <button id="watchlist-close-btn" style="background:none; border:none; cursor:pointer; display:flex; color:#d1d4dc;">{ICON_X}</button>
    </div>
    <button id="scan-watchlist-btn" style="display:flex; align-items:center; justify-content:center; gap:6px; width:100%; padding:6px 0; margin-bottom:10px;">
      {ICON_RADAR} Scan for high-reliability setups
    </button>
    <div id="scan-results"></div>
    <div id="watchlist-list"></div>
  </div>

  <div id="reliability-panel" class="side-panel">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <strong>Pattern reliability</strong>
      <button id="reliability-close-btn" style="background:none; border:none; cursor:pointer; display:flex; color:#d1d4dc;">{ICON_X}</button>
    </div>
    <div id="stats-table"></div>
  </div>

  <div id="oscillators-panel" class="side-panel">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <strong>Oscillators &amp; metrics</strong>
      <button id="oscillators-close-btn" style="background:none; border:none; cursor:pointer; display:flex; color:#d1d4dc;">{ICON_X}</button>
    </div>
    <div id="metrics-dashboard" class="osc-grid"></div>
  </div>

  <div id="strategies-panel" class="side-panel" style="width:520px;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <strong>Strategy leaderboard</strong>
      <button id="strategies-close-btn" style="background:none; border:none; cursor:pointer; display:flex; color:#d1d4dc;">{ICON_X}</button>
    </div>
    <p style="font-size:12px; color:#7d8296; margin:0 0 10px;">
      Every built-in day-trading strategy, backtested on this symbol/interval with identical starting
      capital, ranked by Sharpe ratio. Buy &amp; hold shown as a reference benchmark, not ranked.
    </p>
    <!-- margin-top clears the stacked side-tabs (Watchlist/Reliability/Oscillators/
         Strategies), which are position:fixed at the same right edge as this panel
         and would otherwise sit on top of the table's rightmost columns. -->
    <div id="strategies-leaderboard" style="margin-top:170px;"></div>
  </div>

  <div id="journal-panel" class="side-panel" style="width:760px;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <strong>Trade journal</strong>
      <button id="journal-close-btn" style="background:none; border:none; cursor:pointer; display:flex; color:#d1d4dc;">{ICON_X}</button>
    </div>
    <!-- margin-top clears the stacked side-tabs, same reasoning as
         #strategies-leaderboard -- sized a bit larger since there are now
         5 tabs stacked instead of 4. -->
    <div style="margin-top:180px;">
      <details id="journal-settings-details" style="margin-bottom:10px;">
        <summary style="cursor:pointer; font-size:12px; color:var(--text-dim);">Account settings</summary>
        <div class="journal-form" style="margin-top:8px;">
          <div class="journal-form-row">
            <label>Account balance ($)<input type="number" id="journal-account-balance" step="100"></label>
            <label>Risk per trade (%)<input type="number" id="journal-risk-pct" step="0.25"></label>
            <label>Min R:R ratio<input type="number" id="journal-min-rr" step="0.25"></label>
          </div>
          <button id="journal-save-settings-btn">Save settings</button>
        </div>
      </details>

      <div class="journal-form">
        <div class="journal-form-row">
          <label>Symbol<input type="text" id="journal-symbol" placeholder="AAPL"></label>
          <label>Setup type<input type="text" id="journal-setup-type" placeholder="Breakout"></label>
          <label>Side
            <select id="journal-side"><option value="long">Long</option><option value="short">Short</option></select>
          </label>
        </div>
        <div class="journal-form-row">
          <label>Entry price<input type="number" id="journal-entry" step="0.01"></label>
          <label>Stop loss<input type="number" id="journal-stop" step="0.01"></label>
          <label>Target price<input type="number" id="journal-target" step="0.01"></label>
        </div>
        <div class="journal-form-row">
          <label>Quantity<input type="number" id="journal-quantity" step="1"></label>
          <label style="justify-content:center;" class="journal-checkbox"><input type="checkbox" id="journal-volume-confirmed"> Volume confirmed</label>
          <label style="justify-content:center;" class="journal-checkbox"><input type="checkbox" id="journal-htf-aligned"> Higher TF aligned</label>
        </div>
        <label>Notes<input type="text" id="journal-notes" placeholder="optional"></label>
        <div id="journal-validation"></div>
        <button id="journal-log-trade-btn">{ICON_NOTEBOOK} Log trade</button>
      </div>

      <div id="journal-analytics" class="journal-tiles"></div>

      <div id="journal-filters">
        <select id="journal-filter-status">
          <option value="">All statuses</option>
          <option value="open">Open</option>
          <option value="closed_win">Closed (win)</option>
          <option value="closed_loss">Closed (loss)</option>
          <option value="invalidated">Invalidated</option>
        </select>
        <input type="text" id="journal-filter-setup-type" placeholder="Filter by setup type">
        <input type="text" id="journal-filter-symbol" placeholder="Filter by symbol">
        <button id="journal-export-btn">{ICON_DOWNLOAD} Export CSV</button>
      </div>
      <div style="max-height:320px; overflow:auto;">
        <table style="width:100%; min-width:620px; border-collapse:collapse;">
          <thead><tr style="text-align:left; font-size:10px; color:var(--text-dim); text-transform:uppercase;">
            <th>Symbol</th><th>Setup</th><th>Side</th><th>Entry</th><th>Stop</th><th>Target</th>
            <th>R:R</th><th>P&amp;L</th><th>Status</th><th></th>
          </tr></thead>
          <tbody id="journal-trades-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const SYMBOL_DATA = {json.dumps(data)};
    const PATTERN_NAMES = {json.dumps(ALL_PATTERN_NAMES)};
    const PATTERN_ABBR = {json.dumps(PATTERN_ABBR)};
    const CHART_PATTERN_COLORS = {json.dumps(CHART_PATTERN_COLORS)};
    const MIN_OCCURRENCES_FOR_WIN_RATE = {json.dumps(MIN_OCCURRENCES_FOR_WIN_RATE)};
    const ICON_TRENDING_UP = {json.dumps(ICON_TRENDING_UP)};
    const ICON_TRENDING_DOWN = {json.dumps(ICON_TRENDING_DOWN)};
    const ICON_X = {json.dumps(ICON_X)};
    const ICON_TROPHY = {json.dumps(ICON_TROPHY)};
    const ICON_DOWNLOAD = {json.dumps(ICON_DOWNLOAD)};
    const UP_LABELS = new Set(["Bullish", "Oversold", "Above VWAP"]);
    const DOWN_LABELS = new Set(["Bearish", "Overbought", "Below VWAP"]);
    function directionIcon(label) {{
      if (UP_LABELS.has(label)) return "<span style='color:#26a69a; display:inline-flex;'>" + ICON_TRENDING_UP + "</span>";
      if (DOWN_LABELS.has(label)) return "<span style='color:#ef5350; display:inline-flex;'>" + ICON_TRENDING_DOWN + "</span>";
      return "";
    }}
    let currentSymbol = null;
    let currentInterval = "1h";
    let enabledPatterns = new Set();
    let chartPatternSeriesMap = {{}};  // pattern name -> [series objects] currently drawn, for cleanup
    let linkedCharts = [];  // [{{chart, series, dataByTime}}] -- oscillator-pane crosshair sync group
    let activePanes = {{}};  // paneType -> {{chart, container, linkEntry}}

    // --- Watchlist (persisted server-side in reports/watchlist.json) ------
    let watchlist = {json.dumps(watchlist)};

    // --- On-chart indicators (SMA/EMA/Bollinger/VWAP) ----------------------
    const INDICATOR_COLORS = {{
      SMA: "#f0b90b", EMA: "#ab47bc", BB: "#26c6da", VWAP: "#ec407a",
    }};
    let activeIndicators = [];  // {{id, type, params, series: [...]}}
    let nextIndicatorId = 1;

    // --- Saved chart layout (drawings + indicators, persisted server-side
    // in reports/chart_layouts.json) -----------------------------------
    let restoringLayout = false;  // true while replaying a saved layout, so restore doesn't re-trigger a save

    function saveLayout() {{
      if (!currentSymbol || restoringLayout) return;
      const body = {{
        interval: currentInterval,
        drawings: drawings.map((d) => ({{type: d.type, params: d.params}})),
        indicators: activeIndicators.map((ind) => ({{type: ind.type, params: ind.params}})),
        panes: Object.keys(activePanes),
      }};
      fetch("/api/layout/" + encodeURIComponent(currentSymbol), {{
        method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(body),
      }}).catch(() => {{}});  // static HTML with no server behind it -- fail silently
    }}

    const ET_TIME_FMT = new Intl.DateTimeFormat("en-US", {{timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false}});
    const ET_DAY_FMT = new Intl.DateTimeFormat("en-US", {{timeZone: "America/New_York", month: "short", day: "numeric"}});
    const ET_MONTH_FMT = new Intl.DateTimeFormat("en-US", {{timeZone: "America/New_York", month: "short", year: "numeric"}});
    const ET_YEAR_FMT = new Intl.DateTimeFormat("en-US", {{timeZone: "America/New_York", year: "numeric"}});
    const ET_FULL_FMT = new Intl.DateTimeFormat("en-US", {{timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false}});

    // LightweightCharts tells us each tick's own granularity via
    // tickMarkType (Year/Month/DayOfMonth/Time) -- without switching on it,
    // a formatter that always renders HH:mm shows the same repeated label
    // at any zoom level coarse enough that ticks land on day/week/month
    // boundaries, since every regular session opens at the same wall-clock
    // time (09:30 ET) and every daily bar is stamped at the same time-of-day.
    function formatTickET(time, tickMarkType) {{
      const date = new Date(time * 1000);
      const TMT = LightweightCharts.TickMarkType;
      switch (tickMarkType) {{
        case TMT.Year: return ET_YEAR_FMT.format(date);
        case TMT.Month: return ET_MONTH_FMT.format(date);
        case TMT.DayOfMonth: return ET_DAY_FMT.format(date);
        default: return ET_TIME_FMT.format(date);
      }}
    }}
    const formatFullET = (time) => ET_FULL_FMT.format(new Date(time * 1000)) + " ET";

    const chartEl = document.getElementById("chart");

    // Fills the rest of the viewport below the chart's own top edge, so the
    // chart -- not the header controls above it -- dominates the screen.
    // Recomputed on resize; oscillator panes (if any) stack below and are
    // free to push the page into scroll territory rather than fighting the
    // main chart for a share of one fixed viewport-height budget.
    function computeChartHeight() {{
      const top = chartEl.getBoundingClientRect().top;
      return Math.max(360, Math.round(window.innerHeight - top - 16));
    }}

    const chart = LightweightCharts.createChart(chartEl, {{
      width: chartEl.clientWidth,
      height: computeChartHeight(),
      layout: {{background: {{color: "#0f1117"}}, textColor: "#d1d4dc"}},
      grid: {{vertLines: {{color: "#1c2030"}}, horzLines: {{color: "#1c2030"}}}},
      timeScale: {{timeVisible: true, secondsVisible: false, tickMarkFormatter: formatTickET}},
      localization: {{timeFormatter: formatFullET}},
    }});

    // Solid candlestick bodies, not thin OHLC tick-bars -- this is a
    // *candlestick* pattern dashboard, and thin bars are hard to make out
    // against the dark background, especially zoomed out with many bars
    // packed into a small width.
    const priceSeries = chart.addCandlestickSeries({{
      upColor: "#26a69a", downColor: "#ef5350",
      borderUpColor: "#26a69a", borderDownColor: "#ef5350",
      wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    }});
    const volumeSeries = chart.addHistogramSeries({{priceFormat: {{type: "volume"}}, priceScaleId: ""}});
    volumeSeries.priceScale().applyOptions({{scaleMargins: {{top: 0.8, bottom: 0}}}});
    priceSeries.priceScale().applyOptions({{scaleMargins: {{top: 0.1, bottom: 0.3}}}});

    // Registered into the oscillator-panes crosshair sync group below (main
    // chart is a member too, so hovering a pane moves the main chart's
    // crosshair and vice versa) -- dataByTime is refreshed in loadEntry()
    // each time a new symbol's bars are loaded.
    const mainLinkEntry = registerLinkedChart(chart, priceSeries, new Map());

    // Custom pattern markers -- Lightweight Charts' native setMarkers() only
    // supports a fixed shape enum (circle/square/arrowUp/arrowDown) with no
    // custom shape, so pattern markers are drawn as a synced absolute-
    // positioned DOM overlay instead: a "house" pointer with a 1-3 letter
    // code for candlestick patterns, and start/end labels for chart
    // patterns so overlapping occurrences stay distinguishable.
    chartEl.style.position = "relative";
    const markerOverlay = document.createElement("div");
    markerOverlay.id = "marker-overlay";
    chartEl.appendChild(markerOverlay);
    let overlayMarkers = [];  // {{el, time, price, anchor: "above"|"below"|"mid"}}

    function clearOverlayMarkers() {{
      markerOverlay.innerHTML = "";
      overlayMarkers = [];
    }}

    function addOverlayMarker(el, time, price, anchor) {{
      markerOverlay.appendChild(el);
      overlayMarkers.push({{el: el, time: time, price: price, anchor: anchor}});
    }}

    function positionMarkerList(list) {{
      const timeScale = chart.timeScale();
      list.forEach((om) => {{
        const x = timeScale.timeToCoordinate(om.time);
        const y = priceSeries.priceToCoordinate(om.price);
        if (x === null || y === null) {{ om.el.style.display = "none"; return; }}
        om.el.style.display = "";
        const dx = -om.el.offsetWidth / 2;
        let dy;
        if (om.anchor === "above") dy = -om.el.offsetHeight - 4;
        else if (om.anchor === "below") dy = 4;
        else dy = -om.el.offsetHeight / 2;
        om.el.style.transform = "translate(" + (x + dx) + "px, " + (y + dy) + "px)";
      }});
    }}

    function positionOverlayMarkers() {{
      positionMarkerList(overlayMarkers);
    }}

    // Separate overlay + array from the pattern-marker one above: pattern
    // markers get wholesale rebuilt every time updateMarkers() runs (patterns
    // toggled on/off), which would wipe out user-placed text annotations if
    // they shared the same list.
    const textOverlay = document.createElement("div");
    textOverlay.id = "text-annotation-overlay";
    chartEl.appendChild(textOverlay);
    let textAnnotations = [];  // {{el, time, price, anchor}}

    function positionTextAnnotations() {{
      positionMarkerList(textAnnotations);
    }}

    function houseMarkerEl(text, color, pointDown) {{
      const el = document.createElement("div");
      el.className = "house-marker " + (pointDown ? "house-down" : "house-up");
      el.style.background = color;
      el.textContent = text;
      return el;
    }}

    function chartLabelEl(text, color) {{
      const el = document.createElement("div");
      el.className = "chart-pattern-label";
      el.style.borderColor = color;
      el.style.color = color;
      el.textContent = text;
      return el;
    }}

    function indexBarsByTime(entry) {{
      if (entry._barsByTime) return;
      entry._barsByTime = {{}};
      entry.bars.forEach((b) => {{ entry._barsByTime[b.time] = b; }});
    }}

    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
      positionOverlayMarkers();
      positionTextAnnotations();
      if (range) Object.values(activePanes).forEach((p) => p.chart.timeScale().setVisibleLogicalRange(range));
    }});
    window.addEventListener("resize", () => {{
      chart.applyOptions({{width: chartEl.clientWidth, height: computeChartHeight()}});
      positionOverlayMarkers();
      positionTextAnnotations();
    }});

    chart.subscribeCrosshairMove((param) => {{
      const legend = document.getElementById("ohlc-legend");
      const bar = param.time && param.seriesData ? param.seriesData.get(priceSeries) : null;
      if (!bar) {{ legend.textContent = ""; return; }}
      const vol = param.seriesData.get(volumeSeries);
      let text = "O " + bar.open.toFixed(2) + "  H " + bar.high.toFixed(2) + "  L " + bar.low.toFixed(2) + "  C " + bar.close.toFixed(2);
      if (vol) text += "  Vol " + Math.round(vol.value).toLocaleString();
      legend.textContent = text;
    }});

    // --- Drawing toolbar -----------------------------------------------
    // Drawings are plain LightweightCharts line series / price lines,
    // tracked here so they can be cleanly removed (clear-all, or a symbol
    // switch, which invalidates their time/price coordinates anyway).
    // Each entry also carries its {{type, params}} so the whole set can be
    // serialized to saveLayout() and replayed by restoreLayout().
    let activeTool = "cursor";
    let pendingPoints = [];
    let drawings = [];  // array of {{type, params, remove()}}

    // Per-tool click-by-click guidance shown in #tool-hint -- one entry per
    // click the tool needs, indexed by how many points are already pending
    // so the message advances as the user clicks through a multi-point tool.
    const TOOL_INSTRUCTIONS = {{
      trendline: ["Click the start point", "Click the end point"],
      ray: ["Click the start point", "Click a second point to set the angle -- the line extends past it"],
      hline: ["Click anywhere to place the horizontal line"],
      vline: ["Click anywhere to place the vertical line"],
      channel: ["Click the first point of the base line", "Click the second point of the base line", "Click to set the channel's width"],
      rect: ["Click the first corner", "Click the opposite corner"],
      pitchfork: ["Click the handle (anchor) point", "Click the first fork point", "Click the second fork point"],
      fib: ["Click the swing start", "Click the swing end"],
      fibext: ["Click the swing start", "Click the swing end", "Click the pullback point to project the extension from"],
      fibfan: ["Click the anchor point", "Click a second point to set the fan's angle"],
      text: ["Click where you want to place the note"],
      measure: ["Click the start point", "Click the end point (not saved -- clears when you switch tools)"],
    }};

    function updateToolHint() {{
      const el = document.getElementById("tool-hint");
      const steps = TOOL_INSTRUCTIONS[activeTool];
      if (!steps) {{ el.textContent = ""; return; }}
      const stepIndex = Math.min(pendingPoints.length, steps.length - 1);
      const prefix = steps.length > 1 ? "Step " + (stepIndex + 1) + " of " + steps.length + ": " : "";
      const cancelHint = pendingPoints.length > 0 ? "  (Esc to cancel)" : "";
      el.textContent = prefix + steps[stepIndex] + cancelHint;
    }}

    document.addEventListener("keydown", (e) => {{
      if (e.key === "Escape" && pendingPoints.length > 0) {{
        pendingPoints = [];
        updateToolHint();
      }}
    }});

    function setActiveTool(tool) {{
      activeTool = tool;
      pendingPoints = [];
      clearMeasurement();
      updateToolHint();
      document.querySelectorAll("#drawing-toolbar button[data-tool]").forEach((b) => {{
        b.classList.toggle("tool-btn-active", b.dataset.tool === tool);
      }});
      const showCrosshair = tool !== "cursor";
      chart.applyOptions({{
        crosshair: {{
          vertLine: {{visible: showCrosshair, labelVisible: showCrosshair}},
          horzLine: {{visible: showCrosshair, labelVisible: showCrosshair}},
        }},
      }});
      chartEl.style.cursor = tool === "cursor" ? "default" : "crosshair";
    }}

    function clearDrawings() {{
      drawings.forEach((d) => d.remove());
      drawings = [];
      pendingPoints = [];
    }}

    function addHLineDrawing(price, restoring) {{
      const line = priceSeries.createPriceLine({{
        price: price, color: "#2962ff", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true, title: "",
      }});
      drawings.push({{type: "hline", params: {{price: price}}, remove: () => priceSeries.removePriceLine(line)}});
      if (!restoring) saveLayout();
    }}

    function addTrendlineDrawing(p1, p2, restoring) {{
      const [a, b] = p1.time <= p2.time ? [p1, p2] : [p2, p1];
      const series = chart.addLineSeries({{color: "#2962ff", lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
      series.setData([{{time: a.time, value: a.price}}, {{time: b.time, value: b.price}}]);
      drawings.push({{type: "trendline", params: {{p1: a, p2: b}}, remove: () => chart.removeSeries(series)}});
      if (!restoring) saveLayout();
    }}

    function addChannelDrawing(a, b, p3, restoring) {{
      const lineValueAt = (t) => a.price + (b.price - a.price) * (t - a.time) / (b.time - a.time);
      const offset = p3.price - lineValueAt(p3.time);
      const series1 = chart.addLineSeries({{color: "#2962ff", lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
      series1.setData([{{time: a.time, value: a.price}}, {{time: b.time, value: b.price}}]);
      const series2 = chart.addLineSeries({{
        color: "#2962ff", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false, lastValueVisible: false,
      }});
      series2.setData([{{time: a.time, value: a.price + offset}}, {{time: b.time, value: b.price + offset}}]);
      drawings.push({{
        type: "channel", params: {{a: a, b: b, p3: p3}},
        remove: () => {{ chart.removeSeries(series1); chart.removeSeries(series2); }},
      }});
      if (!restoring) saveLayout();
    }}

    // Extends p1->p2's slope out to just past the last loaded bar, so the
    // ray reads as "this trend line, continued" rather than stopping dead
    // at the second click.
    function rayEndpoint(p1, p2) {{
      const bars = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)].bars;
      const lastTime = bars[bars.length - 1].time;
      const firstTime = bars[0].time;
      const targetTime = Math.max(lastTime + (lastTime - firstTime) * 0.15, p2.time + 1);
      if (p2.time === p1.time) return {{time: targetTime, price: p2.price}};
      const slope = (p2.price - p1.price) / (p2.time - p1.time);
      return {{time: targetTime, price: p1.price + slope * (targetTime - p1.time)}};
    }}

    function addRayDrawing(p1, p2, restoring) {{
      const [a, b] = p1.time <= p2.time ? [p1, p2] : [p2, p1];
      const end = rayEndpoint(a, b);
      const series = chart.addLineSeries({{color: "#2962ff", lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
      series.setData([{{time: a.time, value: a.price}}, {{time: end.time, value: end.price}}]);
      drawings.push({{type: "ray", params: {{p1: a, p2: b}}, remove: () => chart.removeSeries(series)}});
      if (!restoring) saveLayout();
    }}

    // Lightweight Charts' series data is a function of time -- it can't
    // represent a true vertical line (two prices at one time). The
    // established workaround: two points one second apart, which is well
    // under a pixel at any zoom level for minute-or-coarser bars.
    function addVLineDrawing(time, restoring) {{
      const bars = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)].bars;
      const prices = bars.map((b) => b.close);
      const lo = Math.min(...prices), hi = Math.max(...prices);
      const pad = (hi - lo) * 0.5 || 1;
      const series = chart.addLineSeries({{
        color: "#787b86", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false, lastValueVisible: false,
      }});
      series.setData([{{time: time, value: lo - pad}}, {{time: time + 1, value: hi + pad}}]);
      drawings.push({{type: "vline", params: {{time: time}}, remove: () => chart.removeSeries(series)}});
      if (!restoring) saveLayout();
    }}

    // Drawn as 4 edges rather than one closed polyline, since a rectangle's
    // left/right edges are vertical (same time-can't-repeat constraint as
    // addVLineDrawing) and a single series can't close a time-ordered loop.
    function addRectDrawing(p1, p2, restoring) {{
      const [a, b] = p1.time <= p2.time ? [p1, p2] : [p2, p1];
      const top = Math.max(p1.price, p2.price);
      const bottom = Math.min(p1.price, p2.price);
      const mk = () => chart.addLineSeries({{color: "#2962ff", lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
      const topS = mk(); topS.setData([{{time: a.time, value: top}}, {{time: b.time, value: top}}]);
      const botS = mk(); botS.setData([{{time: a.time, value: bottom}}, {{time: b.time, value: bottom}}]);
      const leftS = mk(); leftS.setData([{{time: a.time, value: bottom}}, {{time: a.time + 1, value: top}}]);
      const rightS = mk(); rightS.setData([{{time: b.time, value: bottom}}, {{time: b.time + 1, value: top}}]);
      const series = [topS, botS, leftS, rightS];
      drawings.push({{type: "rect", params: {{p1: a, p2: b}}, remove: () => series.forEach((s) => chart.removeSeries(s))}});
      if (!restoring) saveLayout();
    }}

    const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
    const FIB_COLORS = ["#787b86", "#f23645", "#ff9800", "#4caf50", "#089981", "#2962ff", "#787b86"];

    // Uses priceLine (same mechanism as the horizontal-line tool) rather
    // than a time-bounded series -- fib levels are meant to act as ongoing
    // support/resistance across the whole visible chart, not just the
    // clicked segment, and priceLine's `title` gives each level a label for
    // free.
    function addFibDrawing(p1, p2, restoring) {{
      const lines = FIB_LEVELS.map((level, i) => priceSeries.createPriceLine({{
        price: p1.price + (p2.price - p1.price) * level,
        color: FIB_COLORS[i % FIB_COLORS.length], lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true,
        title: (level * 100).toFixed(1) + "%",
      }}));
      drawings.push({{type: "fib", params: {{p1: p1, p2: p2}}, remove: () => lines.forEach((l) => priceSeries.removePriceLine(l))}});
      if (!restoring) saveLayout();
    }}

    const FIB_EXT_LEVELS = [0, 0.618, 1, 1.272, 1.618, 2, 2.618];

    // 3-point tool: p1->p2 is the measured move, p3 is the pullback/retrace
    // point the extension projects forward from -- standard TradingView Fib
    // Extension semantics.
    function addFibExtDrawing(p1, p2, p3, restoring) {{
      const diff = p2.price - p1.price;
      const lines = FIB_EXT_LEVELS.map((level, i) => priceSeries.createPriceLine({{
        price: p3.price + diff * level,
        color: FIB_COLORS[i % FIB_COLORS.length], lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true,
        title: "Ext " + (level * 100).toFixed(1) + "%",
      }}));
      drawings.push({{
        type: "fibext", params: {{p1: p1, p2: p2, p3: p3}},
        remove: () => lines.forEach((l) => priceSeries.removePriceLine(l)),
      }});
      if (!restoring) saveLayout();
    }}

    const FIB_FAN_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786];

    // 2-point tool: fan lines radiate from p1 through each level's price on
    // the p1->p2 vertical at p2's time, extended onward with the same
    // rayEndpoint() helper the Ray tool uses.
    function addFibFanDrawing(p1, p2, restoring) {{
      const series = FIB_FAN_LEVELS.map((level, i) => {{
        const fanThroughPoint = {{time: p2.time, price: p1.price + (p2.price - p1.price) * level}};
        const end = rayEndpoint(p1, fanThroughPoint);
        const s = chart.addLineSeries({{
          color: FIB_COLORS[i % FIB_COLORS.length], lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
          priceLineVisible: false, lastValueVisible: false,
        }});
        s.setData([{{time: p1.time, value: p1.price}}, {{time: end.time, value: end.price}}]);
        return s;
      }});
      drawings.push({{type: "fibfan", params: {{p1: p1, p2: p2}}, remove: () => series.forEach((s) => chart.removeSeries(s))}});
      if (!restoring) saveLayout();
    }}

    // 3-point tool: p1 is the handle, p2/p3 define the fork width. Median
    // line runs p1 -> midpoint(p2,p3); the two outer tines are parallel to
    // it, passing through p2 and p3 -- all extended with rayEndpoint().
    function addPitchforkDrawing(p1, p2, p3, restoring) {{
      const mid = {{time: (p2.time + p3.time) / 2, price: (p2.price + p3.price) / 2}};
      const medianEnd = rayEndpoint(p1, mid);
      const slope = (mid.price - p1.price) / (mid.time - p1.time);
      const targetTime = medianEnd.time;
      const upperEnd = {{time: targetTime, price: p2.price + slope * (targetTime - p2.time)}};
      const lowerEnd = {{time: targetTime, price: p3.price + slope * (targetTime - p3.time)}};

      const color = "#2962ff";
      const medianS = chart.addLineSeries({{color: color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
      medianS.setData([{{time: p1.time, value: p1.price}}, {{time: medianEnd.time, value: medianEnd.price}}]);
      const upperS = chart.addLineSeries({{color: color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
      upperS.setData([{{time: p2.time, value: p2.price}}, {{time: upperEnd.time, value: upperEnd.price}}]);
      const lowerS = chart.addLineSeries({{color: color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
      lowerS.setData([{{time: p3.time, value: p3.price}}, {{time: lowerEnd.time, value: lowerEnd.price}}]);

      const series = [medianS, upperS, lowerS];
      drawings.push({{type: "pitchfork", params: {{p1: p1, p2: p2, p3: p3}}, remove: () => series.forEach((s) => chart.removeSeries(s))}});
      if (!restoring) saveLayout();
    }}

    function addTextDrawing(point, text, restoring) {{
      const el = document.createElement("div");
      el.className = "text-annotation";
      el.textContent = text;
      textOverlay.appendChild(el);
      const record = {{el: el, time: point.time, price: point.price, anchor: "mid"}};
      textAnnotations.push(record);
      positionTextAnnotations();
      drawings.push({{
        type: "text", params: {{time: point.time, price: point.price, text: text}},
        remove: () => {{ el.remove(); textAnnotations = textAnnotations.filter((t) => t !== record); }},
      }});
      if (!restoring) saveLayout();
    }}

    // --- Measure tool --------------------------------------------------
    // Deliberately NOT part of `drawings` / saveLayout(): a real trading
    // platform's measure tool is a transient "how far is this move" check,
    // not something you'd expect to reopen the chart and still see, unlike
    // every other drawing here.
    let measurementSeries = null;
    let measurementRecord = null;

    function clearMeasurement() {{
      if (measurementSeries) {{ chart.removeSeries(measurementSeries); measurementSeries = null; }}
      if (measurementRecord) {{
        measurementRecord.el.remove();
        textAnnotations = textAnnotations.filter((t) => t !== measurementRecord);
        measurementRecord = null;
      }}
    }}

    function showMeasurement(p1, p2) {{
      clearMeasurement();
      const [a, b] = p1.time <= p2.time ? [p1, p2] : [p2, p1];
      measurementSeries = chart.addLineSeries({{
        color: "#ffca28", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false, lastValueVisible: false,
      }});
      measurementSeries.setData([{{time: a.time, value: a.price}}, {{time: b.time, value: b.price}}]);

      const bars = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)].bars;
      const barCount = bars.filter((bar) => bar.time >= a.time && bar.time <= b.time).length;
      const deltaPrice = p2.price - p1.price;
      const pct = p1.price !== 0 ? (deltaPrice / p1.price) * 100 : 0;
      const sign = deltaPrice >= 0 ? "+" : "";
      const text = sign + deltaPrice.toFixed(2) + " (" + sign + pct.toFixed(2) + "%) · " + barCount + " bars";

      const el = document.createElement("div");
      el.className = "text-annotation measurement-label";
      el.textContent = text;
      textOverlay.appendChild(el);
      measurementRecord = {{el: el, time: b.time, price: b.price, anchor: "mid"}};
      textAnnotations.push(measurementRecord);
      positionTextAnnotations();
    }}

    function closeFlyouts() {{
      document.getElementById("lines-panel").style.display = "none";
      document.getElementById("patterns-panel").style.display = "none";
      document.getElementById("indicators-panel").style.display = "none";
      document.getElementById("backtest-panel").style.display = "none";
    }}

    function toggleFlyout(id) {{
      const el = document.getElementById(id);
      const wasOpen = el.style.display === "block";
      closeFlyouts();
      el.style.display = wasOpen ? "none" : "block";
    }}

    document.getElementById("drawing-toolbar").addEventListener("click", (e) => {{
      const btn = e.target.closest("button[data-tool]");
      if (btn && !btn.disabled) {{
        setActiveTool(btn.dataset.tool);
        closeFlyouts();
      }}
    }});
    document.getElementById("lines-group-trigger").addEventListener("click", (e) => {{
      e.stopPropagation();
      toggleFlyout("lines-panel");
    }});
    document.getElementById("patterns-group-trigger").addEventListener("click", (e) => {{
      e.stopPropagation();
      toggleFlyout("patterns-panel");
    }});
    document.getElementById("indicators-group-trigger").addEventListener("click", (e) => {{
      e.stopPropagation();
      toggleFlyout("indicators-panel");
    }});
    document.getElementById("backtest-group-trigger").addEventListener("click", (e) => {{
      e.stopPropagation();
      toggleFlyout("backtest-panel");
    }});
    document.getElementById("backtest-panel").addEventListener("click", (e) => e.stopPropagation());
    document.getElementById("indicators-panel").addEventListener("click", (e) => {{
      e.stopPropagation();  // clicking a period/stddev input to focus it shouldn't close this flyout
      const btn = e.target.closest("button[data-indicator]");
      if (!btn) return;
      const type = btn.dataset.indicator;
      let params = {{}};
      if (type === "SMA") params = {{period: parseInt(document.getElementById("ind-sma-period").value, 10) || 20}};
      else if (type === "EMA") params = {{period: parseInt(document.getElementById("ind-ema-period").value, 10) || 50}};
      else if (type === "BB") {{
        params = {{
          period: parseInt(document.getElementById("ind-bb-period").value, 10) || 20,
          stddev: parseFloat(document.getElementById("ind-bb-stddev").value) || 2,
        }};
      }}
      if (!currentSymbol) {{ patternHint("Load a symbol first."); return; }}
      addIndicator(type, params, false);
      closeFlyouts();
    }});
    document.addEventListener("click", () => closeFlyouts());
    document.getElementById("clear-drawings-btn").addEventListener("click", () => {{
      clearDrawings();
      saveLayout();
    }});

    // --- Right-edge flyout panels (watchlist, reliability, oscillators) --
    // Mutually exclusive -- they share one backdrop and one screen region,
    // so opening one closes the others rather than stacking on top of them.
    const ALL_SIDE_TABS = ["watchlist-tab", "reliability-tab", "oscillators-tab", "strategies-tab", "journal-tab"];
    function setTabsDimmed(dimmed) {{
      ALL_SIDE_TABS.forEach((id) => document.getElementById(id).classList.toggle("side-tab-dimmed", dimmed));
    }}
    function closeAllPanels() {{
      document.getElementById("watchlist-panel").classList.remove("panel-open");
      document.getElementById("reliability-panel").classList.remove("panel-open");
      document.getElementById("oscillators-panel").classList.remove("panel-open");
      document.getElementById("strategies-panel").classList.remove("panel-open");
      document.getElementById("journal-panel").classList.remove("panel-open");
      document.getElementById("panel-backdrop").classList.remove("backdrop-visible");
      setTabsDimmed(false);
    }}
    function setWatchlistPanel(open) {{
      closeAllPanels();
      if (open) {{
        document.getElementById("watchlist-panel").classList.add("panel-open");
        document.getElementById("panel-backdrop").classList.add("backdrop-visible");
        setTabsDimmed(true);
      }}
    }}
    function setReliabilityPanel(open) {{
      closeAllPanels();
      if (open) {{
        document.getElementById("reliability-panel").classList.add("panel-open");
        document.getElementById("panel-backdrop").classList.add("backdrop-visible");
        setTabsDimmed(true);
      }}
    }}
    function setOscillatorsPanel(open) {{
      closeAllPanels();
      if (open) {{
        document.getElementById("oscillators-panel").classList.add("panel-open");
        document.getElementById("panel-backdrop").classList.add("backdrop-visible");
        setTabsDimmed(true);
      }}
    }}
    function setStrategiesPanel(open) {{
      closeAllPanels();
      if (open) {{
        document.getElementById("strategies-panel").classList.add("panel-open");
        document.getElementById("panel-backdrop").classList.add("backdrop-visible");
        setTabsDimmed(true);
      }}
    }}
    document.getElementById("watchlist-tab").addEventListener("click", () => setWatchlistPanel(true));
    document.getElementById("watchlist-close-btn").addEventListener("click", () => setWatchlistPanel(false));
    document.getElementById("reliability-tab").addEventListener("click", () => setReliabilityPanel(true));
    document.getElementById("reliability-close-btn").addEventListener("click", () => setReliabilityPanel(false));
    document.getElementById("oscillators-tab").addEventListener("click", () => setOscillatorsPanel(true));
    document.getElementById("oscillators-close-btn").addEventListener("click", () => setOscillatorsPanel(false));
    document.getElementById("strategies-tab").addEventListener("click", () => {{ setStrategiesPanel(true); runComparison(); }});
    document.getElementById("strategies-close-btn").addEventListener("click", () => setStrategiesPanel(false));
    function setJournalPanel(open) {{
      closeAllPanels();
      if (open) {{
        document.getElementById("journal-panel").classList.add("panel-open");
        document.getElementById("panel-backdrop").classList.add("backdrop-visible");
        setTabsDimmed(true);
      }}
    }}
    document.getElementById("journal-tab").addEventListener("click", () => {{
      setJournalPanel(true);
      loadJournalSettings();
      loadJournalTrades();
      loadJournalAnalytics();
    }});
    document.getElementById("journal-close-btn").addEventListener("click", () => setJournalPanel(false));
    document.getElementById("panel-backdrop").addEventListener("click", () => closeAllPanels());

    // --- Trade journal -------------------------------------------------
    async function loadJournalSettings() {{
      const resp = await fetch("/api/journal/settings");
      const s = await resp.json();
      document.getElementById("journal-account-balance").value = s.account_balance;
      document.getElementById("journal-risk-pct").value = s.risk_percentage;
      document.getElementById("journal-min-rr").value = s.min_rr_ratio;
    }}
    document.getElementById("journal-save-settings-btn").addEventListener("click", async () => {{
      await fetch("/api/journal/settings", {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          account_balance: parseFloat(document.getElementById("journal-account-balance").value),
          risk_percentage: parseFloat(document.getElementById("journal-risk-pct").value),
          min_rr_ratio: parseFloat(document.getElementById("journal-min-rr").value),
        }}),
      }});
      document.getElementById("journal-settings-details").open = false;
    }});

    let journalValidateTimer = null;
    function scheduleJournalValidate() {{
      clearTimeout(journalValidateTimer);
      journalValidateTimer = setTimeout(runJournalValidate, 250);
    }}
    async function runJournalValidate() {{
      const entry = parseFloat(document.getElementById("journal-entry").value);
      const stop = parseFloat(document.getElementById("journal-stop").value);
      const target = parseFloat(document.getElementById("journal-target").value);
      const box = document.getElementById("journal-validation");
      if (isNaN(entry) || isNaN(stop) || isNaN(target)) {{
        box.className = ""; box.innerHTML = "";
        return;
      }}
      const resp = await fetch("/api/journal/validate", {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          entry_price: entry, stop_loss: stop, target_price: target,
          side: document.getElementById("journal-side").value,
          volume_confirmed: document.getElementById("journal-volume-confirmed").checked,
          higher_tf_aligned: document.getElementById("journal-htf-aligned").checked,
        }}),
      }});
      const v = await resp.json();
      box.className = v.is_valid ? "journal-valid" : "journal-invalid";
      let html = v.rr_ratio !== null ? "<div>R:R = <b>" + v.rr_ratio.toFixed(2) + "</b></div>" : "";
      if (v.suggested_quantity) {{
        html += "<div>Suggested quantity (per risk settings): <b>" + v.suggested_quantity + "</b></div>";
        if (!document.getElementById("journal-quantity").value) {{
          document.getElementById("journal-quantity").value = v.suggested_quantity;
        }}
      }}
      v.errors.forEach((e) => {{ html += '<div class="journal-error">✗ ' + e + "</div>"; }});
      v.warnings.forEach((w) => {{ html += '<div class="journal-warning">⚠ ' + w + "</div>"; }});
      box.innerHTML = html;
    }}
    ["journal-entry", "journal-stop", "journal-target", "journal-side", "journal-volume-confirmed", "journal-htf-aligned"]
      .forEach((id) => document.getElementById(id).addEventListener("input", scheduleJournalValidate));

    document.getElementById("journal-log-trade-btn").addEventListener("click", async () => {{
      const symbol = document.getElementById("journal-symbol").value.trim();
      const setupType = document.getElementById("journal-setup-type").value.trim();
      const quantity = parseInt(document.getElementById("journal-quantity").value, 10);
      if (!symbol || !setupType || !quantity) {{
        alert("Symbol, setup type, and quantity are required.");
        return;
      }}
      const resp = await fetch("/api/journal/trades", {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          symbol, setup_type: setupType, side: document.getElementById("journal-side").value,
          entry_price: parseFloat(document.getElementById("journal-entry").value),
          stop_loss: parseFloat(document.getElementById("journal-stop").value),
          target_price: parseFloat(document.getElementById("journal-target").value),
          quantity, notes: document.getElementById("journal-notes").value,
          volume_confirmed: document.getElementById("journal-volume-confirmed").checked,
          higher_tf_aligned: document.getElementById("journal-htf-aligned").checked,
        }}),
      }});
      const body = await resp.json();
      if (!resp.ok) {{ alert(body.error || "Failed to log trade"); return; }}
      ["journal-symbol", "journal-setup-type", "journal-entry", "journal-stop", "journal-target", "journal-quantity", "journal-notes"]
        .forEach((id) => {{ document.getElementById(id).value = ""; }});
      document.getElementById("journal-volume-confirmed").checked = false;
      document.getElementById("journal-htf-aligned").checked = false;
      document.getElementById("journal-validation").innerHTML = "";
      document.getElementById("journal-validation").className = "";
      loadJournalTrades();
      loadJournalAnalytics();
    }});

    function journalStatsHTML(label, stats) {{
      const pl = stats.net_pl || 0;
      const plColor = pl >= 0 ? "#26a69a" : "#ef5350";
      const winRate = stats.win_rate !== null && stats.win_rate !== undefined ? (stats.win_rate * 100).toFixed(0) + "%" : "--";
      const pf = stats.profit_factor !== null && stats.profit_factor !== undefined ? stats.profit_factor.toFixed(2) : "--";
      return (
        '<div class="journal-tile"><div class="journal-tile-label">' + label + ' Win rate</div><div class="journal-tile-value">' + winRate + "</div></div>" +
        '<div class="journal-tile"><div class="journal-tile-label">' + label + ' Profit factor</div><div class="journal-tile-value">' + pf + "</div></div>" +
        '<div class="journal-tile"><div class="journal-tile-label">' + label + ' Net P&amp;L</div><div class="journal-tile-value" style="color:' + plColor + ';">' + pl.toFixed(2) + "</div></div>" +
        '<div class="journal-tile"><div class="journal-tile-label">' + label + ' Trades</div><div class="journal-tile-value">' + stats.total_trades + "</div></div>"
      );
    }}
    async function loadJournalAnalytics() {{
      const resp = await fetch("/api/journal/analytics");
      const data = await resp.json();
      document.getElementById("journal-analytics").innerHTML = journalStatsHTML("Overall", data.overall);
    }}

    async function journalTradeAction(id, path, options) {{
      const resp = await fetch("/api/journal/trades/" + id + path, options);
      if (!resp.ok) {{ const body = await resp.json(); alert(body.error || "Action failed"); return; }}
      loadJournalTrades();
      loadJournalAnalytics();
    }}
    function renderJournalTrades(trades) {{
      const tbody = document.getElementById("journal-trades-tbody");
      if (!trades.length) {{
        tbody.innerHTML = '<tr><td colspan="10" style="padding:10px 6px; color:var(--text-dim); font-size:12px;">No trades logged yet.</td></tr>';
        return;
      }}
      tbody.innerHTML = trades.map((t) => {{
        const plColor = t.net_pl === null ? "var(--text-dim)" : (t.net_pl >= 0 ? "#26a69a" : "#ef5350");
        const plText = t.net_pl === null ? "--" : t.net_pl.toFixed(2);
        const rrText = t.rr_ratio === null || t.rr_ratio === undefined ? "--" : t.rr_ratio.toFixed(2);
        let actions = "";
        if (t.status === "open") {{
          actions = '<button data-action="close" data-id="' + t.id + '">Close</button>' +
                    '<button data-action="invalidate" data-id="' + t.id + '">Invalidate</button>';
        }}
        actions += '<button data-action="delete" data-id="' + t.id + '">Delete</button>';
        return '<tr class="journal-trade-row">' +
          "<td>" + t.symbol + "</td><td>" + t.setup_type + "</td><td>" + t.side + "</td>" +
          "<td>" + t.entry_price.toFixed(2) + "</td><td>" + t.stop_loss.toFixed(2) + "</td><td>" + t.target_price.toFixed(2) + "</td>" +
          "<td>" + rrText + "</td>" +
          '<td style="color:' + plColor + ';">' + plText + "</td>" +
          "<td>" + t.status + "</td><td>" + actions + "</td></tr>";
      }}).join("");
    }}
    async function loadJournalTrades() {{
      const params = new URLSearchParams();
      const status = document.getElementById("journal-filter-status").value;
      const setupType = document.getElementById("journal-filter-setup-type").value.trim();
      const symbol = document.getElementById("journal-filter-symbol").value.trim();
      if (status) params.set("status", status);
      if (setupType) params.set("setup_type", setupType);
      if (symbol) params.set("symbol", symbol);
      const resp = await fetch("/api/journal/trades?" + params.toString());
      renderJournalTrades(await resp.json());
    }}
    ["journal-filter-status"].forEach((id) => document.getElementById(id).addEventListener("change", loadJournalTrades));
    let journalFilterTimer = null;
    ["journal-filter-setup-type", "journal-filter-symbol"].forEach((id) => {{
      document.getElementById(id).addEventListener("input", () => {{
        clearTimeout(journalFilterTimer);
        journalFilterTimer = setTimeout(loadJournalTrades, 300);
      }});
    }});
    document.getElementById("journal-trades-tbody").addEventListener("click", (e) => {{
      const btn = e.target.closest("button[data-action]");
      if (!btn) return;
      const id = btn.dataset.id;
      if (btn.dataset.action === "close") {{
        const exitPrice = prompt("Exit price:");
        if (exitPrice === null || exitPrice === "") return;
        journalTradeAction(id, "/close", {{
          method: "PATCH", headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{exit_price: parseFloat(exitPrice)}}),
        }});
      }} else if (btn.dataset.action === "invalidate") {{
        journalTradeAction(id, "/invalidate", {{method: "PATCH"}});
      }} else if (btn.dataset.action === "delete") {{
        if (confirm("Delete this trade?")) journalTradeAction(id, "", {{method: "DELETE"}});
      }}
    }});
    document.getElementById("journal-export-btn").addEventListener("click", () => {{
      window.open("/api/journal/export", "_blank");
    }});
    document.getElementById("stats-table").addEventListener("click", (e) => {{
      const cell = e.target.closest(".pattern-glyph-cell");
      if (cell) togglePattern(cell.dataset.patternName);
    }});

    chart.subscribeClick((param) => {{
      if (activeTool === "cursor" || activeTool === "crosshair" || !currentSymbol) return;
      if (!param.point || param.time === undefined) return;
      const price = priceSeries.coordinateToPrice(param.point.y);
      if (price === null || price === undefined) return;
      const point = {{time: param.time, price: price}};

      if (activeTool === "hline") {{
        addHLineDrawing(price, false);
        updateToolHint();
        return;
      }}

      if (activeTool === "vline") {{
        addVLineDrawing(param.time, false);
        updateToolHint();
        return;
      }}

      if (activeTool === "text") {{
        const text = prompt("Annotation text:");
        if (text) addTextDrawing(point, text, false);
        updateToolHint();
        return;
      }}

      if (activeTool === "trendline") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          addTrendlineDrawing(pendingPoints[0], pendingPoints[1], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "ray") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          addRayDrawing(pendingPoints[0], pendingPoints[1], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "rect") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          addRectDrawing(pendingPoints[0], pendingPoints[1], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "fib") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          addFibDrawing(pendingPoints[0], pendingPoints[1], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "fibfan") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          addFibFanDrawing(pendingPoints[0], pendingPoints[1], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "measure") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 2) {{
          showMeasurement(pendingPoints[0], pendingPoints[1]);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "channel") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 3) {{
          const [a, b] = pendingPoints[0].time <= pendingPoints[1].time
            ? [pendingPoints[0], pendingPoints[1]] : [pendingPoints[1], pendingPoints[0]];
          addChannelDrawing(a, b, pendingPoints[2], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "fibext") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 3) {{
          const [a, b] = pendingPoints[0].time <= pendingPoints[1].time
            ? [pendingPoints[0], pendingPoints[1]] : [pendingPoints[1], pendingPoints[0]];
          addFibExtDrawing(a, b, pendingPoints[2], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}

      if (activeTool === "pitchfork") {{
        pendingPoints.push(point);
        if (pendingPoints.length === 3) {{
          addPitchforkDrawing(pendingPoints[0], pendingPoints[1], pendingPoints[2], false);
          pendingPoints = [];
        }}
        updateToolHint();
        return;
      }}
    }});

    function symbolHint(text) {{
      document.getElementById("symbol-search-hint").textContent = text;
    }}

    function patternHint(text) {{
      document.getElementById("pattern-search-hint").textContent = text;
    }}

    // --- Watchlist ---------------------------------------------------------
    function renderWatchlist() {{
      const list = document.getElementById("watchlist-list");
      list.innerHTML = "";
      if (watchlist.length === 0) {{
        list.innerHTML = '<div class="watchlist-empty-hint">No symbols yet -- star the current chart to add it.</div>';
      }}
      watchlist.forEach((symbol) => {{
        const row = document.createElement("div");
        row.className = "watchlist-item" + (symbol === currentSymbol ? " watchlist-item-active" : "");
        const label = document.createElement("span");
        label.textContent = symbol;
        row.appendChild(label);
        const removeBtn = document.createElement("button");
        removeBtn.className = "watchlist-remove-btn";
        removeBtn.innerHTML = ICON_X;
        removeBtn.title = "Remove from watchlist";
        removeBtn.addEventListener("click", (e) => {{ e.stopPropagation(); removeFromWatchlist(symbol); }});
        row.appendChild(removeBtn);
        row.addEventListener("click", () => loadOrFetch(symbol, currentInterval));
        list.appendChild(row);
      }});
      syncWatchlistStarButton();
    }}

    function syncWatchlistStarButton() {{
      const btn = document.getElementById("watchlist-star-btn");
      btn.disabled = !currentSymbol;
      btn.classList.toggle("watchlist-star-active", !!currentSymbol && watchlist.includes(currentSymbol));
    }}

    function addToWatchlist(symbol) {{
      fetch("/api/watchlist", {{
        method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify({{symbol: symbol}}),
      }}).then((r) => r.json()).then((body) => {{
        watchlist = body.symbols;
        renderWatchlist();
      }}).catch(() => {{}});
    }}

    function removeFromWatchlist(symbol) {{
      fetch("/api/watchlist/" + encodeURIComponent(symbol), {{method: "DELETE"}})
        .then((r) => r.json()).then((body) => {{
          watchlist = body.symbols;
          renderWatchlist();
        }}).catch(() => {{}});
    }}

    document.getElementById("watchlist-star-btn").addEventListener("click", () => {{
      if (!currentSymbol) return;
      if (watchlist.includes(currentSymbol)) removeFromWatchlist(currentSymbol);
      else addToWatchlist(currentSymbol);
    }});

    // --- Watchlist scanner --------------------------------------------
    // Cross-references "what's actively happening on this symbol right
    // now" (recently confirmed patterns) against "how reliable has this
    // exact pattern been on this exact symbol historically" -- both
    // already computed per-symbol elsewhere in this dashboard, just never
    // run across the whole watchlist at once before.
    const DIRECTION_HIT_COLOR = {{bullish: "#26a69a", bearish: "#ef5350", neutral: "#7d8296"}};

    function renderScanResults(data) {{
      const container = document.getElementById("scan-results");
      if (!data.hits.length) {{
        container.innerHTML = '<div class="scan-hint">No high-reliability setups active right now (scanned ' +
          data.scanned + ' symbol' + (data.scanned === 1 ? "" : "s") + ').</div>';
        return;
      }}
      container.innerHTML = "";
      data.hits.forEach((hit) => {{
        const color = DIRECTION_HIT_COLOR[hit.direction] || "#7d8296";
        const row = document.createElement("div");
        row.className = "scan-hit";
        row.style.borderColor = color;
        const label = hit.pattern_name.replace(/_/g, " ");
        row.innerHTML =
          '<div class="scan-hit-left">' +
            '<span class="scan-hit-symbol">' + hit.symbol + '</span>' +
            '<span class="scan-hit-pattern">' + label + ' (n=' + hit.occurrences + ')</span>' +
          '</div>' +
          '<span class="scan-hit-winrate" style="color:' + color + ';">' + (hit.win_rate * 100).toFixed(0) + '%</span>';
        row.addEventListener("click", () => loadOrFetch(hit.symbol, currentInterval));
        container.appendChild(row);
      }});
    }}

    async function runScan() {{
      const btn = document.getElementById("scan-watchlist-btn");
      const container = document.getElementById("scan-results");
      if (watchlist.length === 0) {{
        container.innerHTML = '<div class="scan-hint">Star a symbol first -- nothing to scan yet.</div>';
        return;
      }}
      const originalHTML = btn.innerHTML;
      btn.disabled = true;
      btn.textContent = "Scanning " + watchlist.length + " symbol" + (watchlist.length === 1 ? "" : "s") + "...";
      container.innerHTML = "";
      try {{
        const resp = await fetch("/api/scan", {{
          method: "POST", headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{interval: currentInterval || "1h"}}),
        }});
        const body = await resp.json();
        if (!resp.ok) {{
          container.innerHTML = '<div class="scan-hint">' + (body.error || "Scan failed") + '</div>';
          return;
        }}
        renderScanResults(body);
      }} catch (err) {{
        container.innerHTML = '<div class="scan-hint">Scan request failed: ' + err + '</div>';
      }} finally {{
        btn.disabled = false;
        btn.innerHTML = originalHTML;
      }}
    }}
    document.getElementById("scan-watchlist-btn").addEventListener("click", runScan);

    function renderActivePatternTags() {{
      const container = document.getElementById("active-patterns");
      container.innerHTML = "";
      enabledPatterns.forEach((name) => {{
        const tag = document.createElement("span");
        tag.innerHTML = name + " <span style='display:inline-flex; vertical-align:middle;'>" + ICON_X + "</span>";
        tag.style.cssText = "display:inline-flex; align-items:center; gap:4px; background:#262a37; border:1px solid #2f3444; border-radius:4px; padding:3px 8px; margin:2px; cursor:pointer; font-size:13px; color:#d1d4dc;";
        tag.addEventListener("click", () => togglePattern(name));
        container.appendChild(tag);
      }});
    }}

    function renderChartPatternLines(name) {{
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      const occurrences = (entry.chart_patterns && entry.chart_patterns[name]) || [];
      const color = CHART_PATTERN_COLORS[name] || "#7d8296";
      const seriesList = [];
      occurrences.forEach((occ) => {{
        const vertexSeries = chart.addLineSeries({{color: color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
        vertexSeries.setData(occ.vertices.map(v => ({{time: v.time, value: v.value}})));
        seriesList.push(vertexSeries);

        occ.trendlines.forEach((line) => {{
          const lineSeries = chart.addLineSeries({{
            color: color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false, lastValueVisible: false,
          }});
          lineSeries.setData(line.map(v => ({{time: v.time, value: v.value}})));
          seriesList.push(lineSeries);
        }});
      }});
      chartPatternSeriesMap[name] = seriesList;
    }}

    function removeChartPatternLines(name) {{
      (chartPatternSeriesMap[name] || []).forEach(s => chart.removeSeries(s));
      delete chartPatternSeriesMap[name];
    }}

    // --- On-chart indicators -----------------------------------------------
    // Computed client-side from entry.bars (already fully shipped to the
    // browser for the candlestick series) rather than round-tripping to the
    // server -- lets period/stddev be tweaked with no extra fetch.
    function computeSMA(bars, period) {{
      const out = [];
      let sum = 0;
      for (let i = 0; i < bars.length; i++) {{
        sum += bars[i].close;
        if (i >= period) sum -= bars[i - period].close;
        if (i >= period - 1) out.push({{time: bars[i].time, value: sum / period}});
      }}
      return out;
    }}

    function computeEMA(bars, period) {{
      const out = [];
      const k = 2 / (period + 1);
      let prev = null;
      bars.forEach((b) => {{
        prev = prev === null ? b.close : b.close * k + prev * (1 - k);
        out.push({{time: b.time, value: prev}});
      }});
      return out;
    }}

    function computeBollinger(bars, period, stddev) {{
      const mid = [], upper = [], lower = [];
      for (let i = 0; i < bars.length; i++) {{
        if (i < period - 1) continue;
        const window = bars.slice(i - period + 1, i + 1).map(b => b.close);
        const mean = window.reduce((a, v) => a + v, 0) / period;
        const variance = window.reduce((a, v) => a + (v - mean) * (v - mean), 0) / period;
        const sd = Math.sqrt(variance);
        mid.push({{time: bars[i].time, value: mean}});
        upper.push({{time: bars[i].time, value: mean + stddev * sd}});
        lower.push({{time: bars[i].time, value: mean - stddev * sd}});
      }}
      return {{mid: mid, upper: upper, lower: lower}};
    }}

    function computeSessionVWAP(bars) {{
      const out = [];
      let day = null, cumPV = 0, cumVol = 0;
      bars.forEach((b) => {{
        const d = new Date(b.time * 1000).toISOString().slice(0, 10);
        if (d !== day) {{ day = d; cumPV = 0; cumVol = 0; }}
        const typical = (b.high + b.low + b.close) / 3;
        cumPV += typical * b.volume;
        cumVol += b.volume;
        out.push({{time: b.time, value: cumVol > 0 ? cumPV / cumVol : b.close}});
      }});
      return out;
    }}

    // --- Oscillator pane math (mirrors stockx/strategies/indicators.py) ---
    function computeRSI(bars, period) {{
      period = period || 14;
      const n = bars.length;
      const gains = new Array(n).fill(0);
      const losses = new Array(n).fill(0);
      for (let i = 1; i < n; i++) {{
        const delta = bars[i].close - bars[i - 1].close;
        gains[i] = Math.max(delta, 0);
        losses[i] = Math.max(-delta, 0);
      }}
      const alpha = 1 / period;
      const avgGain = new Array(n), avgLoss = new Array(n);
      avgGain[0] = gains[0];
      avgLoss[0] = losses[0];
      for (let i = 1; i < n; i++) {{
        avgGain[i] = (1 - alpha) * avgGain[i - 1] + alpha * gains[i];
        avgLoss[i] = (1 - alpha) * avgLoss[i - 1] + alpha * losses[i];
      }}
      const out = [];
      for (let i = period - 1; i < n; i++) {{
        const value = avgLoss[i] === 0 ? 100 : 100 - 100 / (1 + avgGain[i] / avgLoss[i]);
        out.push({{time: bars[i].time, value: value}});
      }}
      return out;
    }}

    function computeMACD(bars, fast, slow, signal) {{
      fast = fast || 12; slow = slow || 26; signal = signal || 9;
      const emaFast = computeEMA(bars, fast).map((p) => p.value);
      const emaSlow = computeEMA(bars, slow).map((p) => p.value);
      const macdVals = emaFast.map((v, i) => v - emaSlow[i]);
      const k = 2 / (signal + 1);
      const signalVals = [];
      let prev = null;
      macdVals.forEach((v) => {{
        prev = prev === null ? v : v * k + prev * (1 - k);
        signalVals.push(prev);
      }});
      const macdLine = [], signalLine = [], histogram = [];
      bars.forEach((b, i) => {{
        const h = macdVals[i] - signalVals[i];
        macdLine.push({{time: b.time, value: macdVals[i]}});
        signalLine.push({{time: b.time, value: signalVals[i]}});
        histogram.push({{time: b.time, value: h, color: h >= 0 ? "#26a69a" : "#ef5350"}});
      }});
      return {{macd: macdLine, signal: signalLine, histogram: histogram}};
    }}

    function computeStochastic(bars, kPeriod, dPeriod) {{
      kPeriod = kPeriod || 14; dPeriod = dPeriod || 3;
      const n = bars.length;
      const kVals = new Array(n).fill(null);
      for (let i = kPeriod - 1; i < n; i++) {{
        let lowMin = Infinity, highMax = -Infinity;
        for (let j = i - kPeriod + 1; j <= i; j++) {{
          lowMin = Math.min(lowMin, bars[j].low);
          highMax = Math.max(highMax, bars[j].high);
        }}
        const range = highMax - lowMin;
        kVals[i] = range === 0 ? null : 100 * (bars[i].close - lowMin) / range;
      }}
      const percentK = [], percentD = [];
      for (let i = 0; i < n; i++) {{
        if (kVals[i] !== null) percentK.push({{time: bars[i].time, value: kVals[i]}});
      }}
      for (let i = kPeriod - 1 + dPeriod - 1; i < n; i++) {{
        let sum = 0, ok = true;
        for (let j = i - dPeriod + 1; j <= i; j++) {{
          if (kVals[j] === null) {{ ok = false; break; }}
          sum += kVals[j];
        }}
        if (ok) percentD.push({{time: bars[i].time, value: sum / dPeriod}});
      }}
      return {{k: percentK, d: percentD}};
    }}

    function indicatorLabel(type, params) {{
      if (type === "SMA") return "SMA(" + params.period + ")";
      if (type === "EMA") return "EMA(" + params.period + ")";
      if (type === "BB") return "BB(" + params.period + "," + params.stddev + ")";
      return "VWAP";
    }}

    function addIndicator(type, params, restoring) {{
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      const bars = entry.bars;  // {{time, open, high, low, close}}
      const color = INDICATOR_COLORS[type] || "#7d8296";
      const series = [];

      if (type === "SMA") {{
        const s = chart.addLineSeries({{color: color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
        s.setData(computeSMA(bars, params.period));
        series.push(s);
      }} else if (type === "EMA") {{
        const s = chart.addLineSeries({{color: color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
        s.setData(computeEMA(bars, params.period));
        series.push(s);
      }} else if (type === "BB") {{
        const bands = computeBollinger(bars, params.period, params.stddev);
        const upperS = chart.addLineSeries({{color: color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
        upperS.setData(bands.upper);
        const midS = chart.addLineSeries({{color: color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false}});
        midS.setData(bands.mid);
        const lowerS = chart.addLineSeries({{color: color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
        lowerS.setData(bands.lower);
        series.push(upperS, midS, lowerS);
      }} else if (type === "VWAP") {{
        const s = chart.addLineSeries({{color: color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
        s.setData(computeSessionVWAP(bars));
        series.push(s);
      }}

      activeIndicators.push({{id: nextIndicatorId++, type: type, params: params, series: series}});
      renderActiveIndicatorTags();
      if (!restoring) saveLayout();
    }}

    function removeIndicator(id) {{
      const ind = activeIndicators.find((i) => i.id === id);
      if (!ind) return;
      ind.series.forEach((s) => chart.removeSeries(s));
      activeIndicators = activeIndicators.filter((i) => i.id !== id);
      renderActiveIndicatorTags();
      saveLayout();
    }}

    function clearIndicators() {{
      activeIndicators.forEach((ind) => ind.series.forEach((s) => chart.removeSeries(s)));
      activeIndicators = [];
      renderActiveIndicatorTags();
    }}

    function renderActiveIndicatorTags() {{
      const container = document.getElementById("active-indicators");
      container.innerHTML = "";
      activeIndicators.forEach((ind) => {{
        const tag = document.createElement("span");
        const color = INDICATOR_COLORS[ind.type] || "#7d8296";
        tag.innerHTML = indicatorLabel(ind.type, ind.params) + " <span style='display:inline-flex; vertical-align:middle;'>" + ICON_X + "</span>";
        tag.style.cssText = "display:inline-flex; align-items:center; gap:4px; background:#262a37; border:1px solid " + color +
          "; color:" + color + "; border-radius:4px; padding:3px 8px; margin:2px; cursor:pointer; font-size:13px;";
        tag.addEventListener("click", () => removeIndicator(ind.id));
        container.appendChild(tag);
      }});
    }}

    // --- Oscillator panes (RSI / MACD / Stochastic) ------------------------
    // Lightweight Charts v4 has no native stacked-pane API (that's v5-only,
    // and v5 renamed series creation entirely) -- so each pane is an
    // independent chart instance stacked under the main one, kept in sync
    // via linkedCharts (crosshair) and a direct visible-range broadcast from
    // the main chart (panes have handleScroll/handleScale disabled, so
    // there's no reverse direction to reconcile).
    function registerLinkedChart(chartInstance, series, dataByTime) {{
      const entry = {{chart: chartInstance, series: series, dataByTime: dataByTime}};
      linkedCharts.push(entry);
      chartInstance.subscribeCrosshairMove((param) => {{
        if (!param.time) {{
          linkedCharts.forEach((other) => {{ if (other !== entry) other.chart.clearCrosshairPosition(); }});
          return;
        }}
        linkedCharts.forEach((other) => {{
          if (other === entry) return;
          const val = other.dataByTime.get(param.time);
          if (val !== undefined) other.chart.setCrosshairPosition(val, param.time, other.series);
          else other.chart.clearCrosshairPosition();
        }});
      }});
      return entry;
    }}

    function unregisterLinkedChart(entry) {{
      linkedCharts = linkedCharts.filter((e) => e !== entry);
    }}

    function seriesToDataByTime(points) {{
      const m = new Map();
      points.forEach((p) => m.set(p.time, p.value));
      return m;
    }}

    function syncPaneButton(type) {{
      const btn = document.querySelector('.pane-toggle-btn[data-pane="' + type + '"]');
      if (btn) btn.classList.toggle("tool-btn-active", !!activePanes[type]);
    }}

    const FIXED_0_100 = () => ({{priceRange: {{minValue: 0, maxValue: 100}}}});

    function makePaneChart(label, container) {{
      container = container || document.getElementById("oscillator-panes");
      const wrapper = document.createElement("div");
      wrapper.className = "oscillator-pane";
      const labelEl = document.createElement("div");
      labelEl.className = "oscillator-pane-label";
      labelEl.textContent = label;
      const chartDiv = document.createElement("div");
      chartDiv.style.width = "100%";
      wrapper.appendChild(labelEl);
      wrapper.appendChild(chartDiv);
      container.appendChild(wrapper);

      const paneChart = LightweightCharts.createChart(chartDiv, {{
        width: chartDiv.clientWidth,
        height: 120,
        layout: {{background: {{color: "#0f1117"}}, textColor: "#d1d4dc"}},
        grid: {{vertLines: {{color: "#1c2030"}}, horzLines: {{color: "#1c2030"}}}},
        timeScale: {{timeVisible: true, secondsVisible: false, tickMarkFormatter: formatTickET}},
        localization: {{timeFormatter: formatFullET}},
        handleScroll: false, handleScale: false,
      }});
      const mainRange = chart.timeScale().getVisibleLogicalRange();
      if (mainRange) paneChart.timeScale().setVisibleLogicalRange(mainRange);
      return {{wrapper: wrapper, chartDiv: chartDiv, paneChart: paneChart}};
    }}

    function createPane(type, restoring) {{
      if (activePanes[type] || !currentSymbol) return;
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      const bars = entry.bars;
      let built, primarySeries, dataByTime;

      if (type === "RSI") {{
        built = makePaneChart("RSI(14)");
        const s = built.paneChart.addLineSeries({{
          color: "#42a5f5", lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
          autoscaleInfoProvider: FIXED_0_100,
        }});
        const data = computeRSI(bars, 14);
        s.setData(data);
        s.createPriceLine({{price: 70, color: "#7d8296", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true}});
        s.createPriceLine({{price: 30, color: "#7d8296", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true}});
        primarySeries = s;
        dataByTime = seriesToDataByTime(data);
      }} else if (type === "Stoch") {{
        built = makePaneChart("Stochastic(14,3)");
        const kSeries = built.paneChart.addLineSeries({{
          color: "#42a5f5", lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
          autoscaleInfoProvider: FIXED_0_100,
        }});
        const dSeries = built.paneChart.addLineSeries({{
          color: "#ffca28", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
          priceLineVisible: false, lastValueVisible: false, autoscaleInfoProvider: FIXED_0_100,
        }});
        const {{k, d}} = computeStochastic(bars, 14, 3);
        kSeries.setData(k);
        dSeries.setData(d);
        kSeries.createPriceLine({{price: 80, color: "#7d8296", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true}});
        kSeries.createPriceLine({{price: 20, color: "#7d8296", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true}});
        primarySeries = kSeries;
        dataByTime = seriesToDataByTime(k);
      }} else if (type === "MACD") {{
        built = makePaneChart("MACD(12,26,9)");
        const histSeries = built.paneChart.addHistogramSeries({{priceLineVisible: false, lastValueVisible: false}});
        const macdSeries = built.paneChart.addLineSeries({{color: "#f0b90b", lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
        const signalSeries = built.paneChart.addLineSeries({{color: "#ab47bc", lineWidth: 1, priceLineVisible: false, lastValueVisible: false}});
        const {{macd, signal, histogram}} = computeMACD(bars, 12, 26, 9);
        histSeries.setData(histogram);
        macdSeries.setData(macd);
        signalSeries.setData(signal);
        macdSeries.createPriceLine({{price: 0, color: "#7d8296", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false}});
        primarySeries = macdSeries;
        dataByTime = seriesToDataByTime(macd);
      }} else {{
        return;
      }}

      const linkEntry = registerLinkedChart(built.paneChart, primarySeries, dataByTime);
      activePanes[type] = {{chart: built.paneChart, container: built.wrapper, linkEntry: linkEntry}};
      syncPaneButton(type);
      if (!restoring) saveLayout();
    }}

    function destroyPane(type) {{
      const p = activePanes[type];
      if (!p) return;
      unregisterLinkedChart(p.linkEntry);
      p.chart.remove();
      p.container.remove();
      delete activePanes[type];
      syncPaneButton(type);
    }}

    function clearAllPanes() {{
      Object.keys(activePanes).forEach((type) => destroyPane(type));
    }}

    document.querySelectorAll(".pane-toggle-btn").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const type = btn.dataset.pane;
        if (activePanes[type]) {{
          destroyPane(type);
          saveLayout();
        }} else {{
          if (!currentSymbol) {{ patternHint("Load a symbol first."); return; }}
          createPane(type, false);
        }}
      }});
    }});

    // --- Backtesting ---------------------------------------------------
    // Deliberately NOT part of saveLayout()/applyLayout(): a backtest run
    // is a point-in-time computation over whatever data currently exists,
    // not a stable chart setting -- same "ephemeral" treatment as Measure.
    let strategiesList = [];
    let backtestPaneChart = null;
    let backtestPaneWrapper = null;
    let backtestLinkEntry = null;

    function metricPct(v) {{ return v === null || v === undefined ? "N/A" : (v * 100).toFixed(2) + "%"; }}
    function metricNum(v, decimals) {{ return v === null || v === undefined ? "N/A" : v.toFixed(decimals === undefined ? 2 : decimals); }}

    async function loadStrategiesList() {{
      if (strategiesList.length) return strategiesList;
      try {{
        const resp = await fetch("/api/strategies");
        strategiesList = await resp.json();
      }} catch (err) {{ strategiesList = []; }}
      return strategiesList;
    }}

    function renderBacktestParamsForm(strategyName) {{
      const strategy = strategiesList.find((s) => s.name === strategyName);
      const container = document.getElementById("backtest-params-form");
      container.innerHTML = "";
      if (!strategy) return;
      const choices = strategy.param_choices || {{}};
      Object.entries(strategy.default_params).forEach(([key, value]) => {{
        const label = document.createElement("label");
        if (choices[key]) {{
          const options = choices[key].map((c) =>
            '<option value="' + c + '"' + (c === value ? " selected" : "") + ">" + c + "</option>"
          ).join("");
          label.innerHTML = key + ' <select data-param="' + key + '" data-param-type="string">' + options + "</select>";
        }} else {{
          const step = Number.isInteger(value) ? "1" : "0.1";
          label.innerHTML = key + ' <input type="number" data-param="' + key + '" value="' + value + '" step="' + step + '">';
        }}
        container.appendChild(label);
      }});
    }}

    async function populateBacktestStrategySelect() {{
      await loadStrategiesList();
      const select = document.getElementById("backtest-strategy-select");
      select.innerHTML = strategiesList.map((s) => '<option value="' + s.name + '">' + s.display_name + "</option>").join("");
      if (strategiesList.length) renderBacktestParamsForm(strategiesList[0].name);
    }}

    document.getElementById("backtest-strategy-select").addEventListener("change", (e) => {{
      renderBacktestParamsForm(e.target.value);
    }});

    function clearBacktest() {{
      if (backtestLinkEntry) {{ unregisterLinkedChart(backtestLinkEntry); backtestLinkEntry = null; }}
      if (backtestPaneChart) {{ backtestPaneChart.remove(); backtestPaneChart = null; }}
      if (backtestPaneWrapper) {{ backtestPaneWrapper.remove(); backtestPaneWrapper = null; }}
      priceSeries.setMarkers([]);
      document.getElementById("backtest-results").innerHTML = "";
    }}

    function renderBacktestSummary(data) {{
      const container = document.getElementById("backtest-results");
      const m = data.metrics;
      const returnColor = (m.total_return || 0) >= 0 ? "#26a69a" : "#ef5350";
      const summary = document.createElement("div");
      summary.id = "backtest-summary";
      summary.style.cssText = "padding:6px 8px; font-size:12px; color:#d1d4dc; display:flex; flex-wrap:wrap; " +
        "gap:14px; align-items:center; border-top:1px solid #2f3444; background:#12141c;";
      summary.innerHTML =
        "<strong style='color:#fff;'>" + data.display_name + "</strong>" +
        "<span>Return <b style='color:" + returnColor + ";'>" + metricPct(m.total_return) + "</b></span>" +
        "<span>Sharpe <b>" + metricNum(m.sharpe_ratio) + "</b></span>" +
        "<span>Sortino <b>" + metricNum(m.sortino_ratio) + "</b></span>" +
        "<span>Win rate <b>" + metricPct(m.win_rate) + "</b></span>" +
        "<span>Max DD <b style='color:#ef5350;'>" + metricPct(m.max_drawdown) + "</b></span>" +
        "<span>Trades <b>" + m.num_trades + "</b></span>" +
        '<button id="backtest-clear-btn" style="margin-left:auto; font-size:11px; padding:2px 8px; display:flex; align-items:center; gap:4px;">' + ICON_X + " Clear</button>";
      container.appendChild(summary);
      document.getElementById("backtest-clear-btn").addEventListener("click", clearBacktest);
    }}

    function renderBacktestEquityPane(data) {{
      const container = document.getElementById("backtest-results");
      const built = makePaneChart("Equity curve -- " + data.display_name, container);
      const series = built.paneChart.addLineSeries({{color: "#26a69a", lineWidth: 2, priceLineVisible: false, lastValueVisible: false}});
      series.setData(data.equity_curve);
      backtestPaneChart = built.paneChart;
      backtestPaneWrapper = built.wrapper;
      backtestLinkEntry = registerLinkedChart(built.paneChart, series, seriesToDataByTime(data.equity_curve));
    }}

    // Both the chart markers and the trades table/CSV are derived from this
    // one canonical per-trade row shape (entry/exit time+price, side, qty,
    // pnl, return_pct, commission, slippage) -- no separate marker payload
    // to keep in sync.
    function tradeMarkersToLWC(trades) {{
      const markers = [];
      trades.forEach((t) => {{
        markers.push(t.side === 1
          ? {{time: t.entry_time, position: "belowBar", color: "#26a69a", shape: "arrowUp", text: "Buy"}}
          : {{time: t.entry_time, position: "aboveBar", color: "#ef5350", shape: "arrowDown", text: "Short"}});
        const pnlColor = t.pnl >= 0 ? "#26a69a" : "#ef5350";
        markers.push(t.side === 1
          ? {{time: t.exit_time, position: "aboveBar", color: pnlColor, shape: "arrowDown", text: "Sell"}}
          : {{time: t.exit_time, position: "belowBar", color: pnlColor, shape: "arrowUp", text: "Cover"}});
      }});
      return markers.sort((a, b) => a.time - b.time);
    }}

    function formatTradeTime(ts) {{
      const date = new Date(ts * 1000);
      return ET_DAY_FMT.format(date) + " " + ET_TIME_FMT.format(date);
    }}

    function zoomChartToTrade(t) {{
      const span = Math.max(t.exit_time - t.entry_time, 60);
      const pad = Math.max(span * 0.6, 3600);
      chart.timeScale().setVisibleRange({{from: t.entry_time - pad, to: t.exit_time + pad}});
      document.getElementById("chart").scrollIntoView({{behavior: "smooth", block: "center"}});
    }}

    function downloadTradesCSV(data) {{
      const headers = ["#", "Entry Time (ET)", "Entry Price", "Exit Time (ET)", "Exit Price", "Side", "Qty", "PnL", "Return %", "Commission", "Slippage"];
      const rows = data.trades.map((t, i) => [
        i + 1, formatTradeTime(t.entry_time), t.entry_price, formatTradeTime(t.exit_time), t.exit_price,
        t.side === 1 ? "Long" : "Short", t.qty, t.pnl, (t.return_pct * 100).toFixed(4), t.commission, t.slippage,
      ]);
      const csv = [headers, ...rows].map((r) => r.map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
      const blob = new Blob([csv], {{type: "text/csv"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.symbol + "_" + data.strategy + "_trades.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    function renderBacktestTradesTable(data) {{
      const container = document.getElementById("backtest-results");
      const wrapper = document.createElement("div");
      wrapper.id = "backtest-trades-panel";
      wrapper.style.cssText = "border-top:1px solid #2f3444;";

      const header = document.createElement("div");
      header.style.cssText = "display:flex; align-items:center; justify-content:space-between; padding:6px 8px; background:#12141c;";
      header.innerHTML =
        '<span style="font-size:11px; color:#7d8296; font-family:monospace;">Trades (' + data.trades.length + ')</span>' +
        '<button id="backtest-csv-btn" style="font-size:11px; padding:2px 8px; display:flex; align-items:center; gap:4px;" ' +
        (data.trades.length ? "" : "disabled") + '>' + ICON_DOWNLOAD + ' Download CSV</button>';
      wrapper.appendChild(header);

      if (!data.trades.length) {{
        const empty = document.createElement("div");
        empty.style.cssText = "padding:10px 8px; font-size:12px; color:#7d8296;";
        empty.textContent = "This strategy took no trades over this data range.";
        wrapper.appendChild(empty);
        container.appendChild(wrapper);
        return;
      }}

      const tableWrap = document.createElement("div");
      tableWrap.style.cssText = "max-height:280px; overflow-y:auto;";
      const rows = data.trades.map((t, i) => {{
        const pnlColor = t.pnl >= 0 ? "#26a69a" : "#ef5350";
        return '<tr class="backtest-trade-row" data-idx="' + i + '" style="cursor:pointer;">' +
          "<td>" + (i + 1) + "</td>" +
          "<td>" + formatTradeTime(t.entry_time) + "</td>" +
          "<td>" + t.entry_price.toFixed(2) + "</td>" +
          "<td>" + formatTradeTime(t.exit_time) + "</td>" +
          "<td>" + t.exit_price.toFixed(2) + "</td>" +
          "<td>" + (t.side === 1 ? "Long" : "Short") + "</td>" +
          "<td>" + t.qty + "</td>" +
          '<td style="color:' + pnlColor + ';">' + t.pnl.toFixed(2) + "</td>" +
          '<td style="color:' + pnlColor + ';">' + (t.return_pct * 100).toFixed(2) + "%</td>" +
        "</tr>";
      }}).join("");
      tableWrap.innerHTML =
        '<table style="width:100%; font-size:12px;">' +
        "<tr><th>#</th><th>Entry</th><th>Entry Px</th><th>Exit</th><th>Exit Px</th><th>Side</th><th>Qty</th><th>P&amp;L</th><th>Return</th></tr>" +
        rows +
        "</table>";
      wrapper.appendChild(tableWrap);
      container.appendChild(wrapper);

      document.getElementById("backtest-csv-btn").addEventListener("click", () => downloadTradesCSV(data));
      tableWrap.querySelectorAll(".backtest-trade-row").forEach((row) => {{
        row.addEventListener("click", () => zoomChartToTrade(data.trades[parseInt(row.dataset.idx, 10)]));
      }});
    }}

    function renderBacktestResult(data) {{
      clearBacktest();
      renderBacktestSummary(data);
      renderBacktestEquityPane(data);
      renderBacktestTradesTable(data);
      priceSeries.setMarkers(tradeMarkersToLWC(data.trades));
    }}

    async function runBacktest() {{
      if (!currentSymbol) {{ patternHint("Load a symbol first."); return; }}
      const strategyName = document.getElementById("backtest-strategy-select").value;
      const executionTiming = document.getElementById("backtest-execution-timing-select").value;
      const intrabarPath = document.getElementById("backtest-intrabar-path-select").value;
      const params = {{}};
      document.querySelectorAll("#backtest-params-form [data-param]").forEach((input) => {{
        params[input.dataset.param] = input.dataset.paramType === "string" ? input.value : parseFloat(input.value);
      }});
      const btn = document.getElementById("backtest-run-btn");
      const originalHTML = btn.innerHTML;
      btn.disabled = true;
      btn.textContent = "Running...";
      try {{
        const resp = await fetch("/api/backtest/" + encodeURIComponent(currentSymbol), {{
          method: "POST", headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            interval: currentInterval, strategy: strategyName, params: params,
            execution_timing: executionTiming, intrabar_path: intrabarPath,
          }}),
        }});
        const body = await resp.json();
        if (!resp.ok) {{ alert(body.error || "Backtest failed"); return; }}
        renderBacktestResult(body);
      }} catch (err) {{
        alert("Backtest request failed: " + err);
      }} finally {{
        btn.disabled = false;
        btn.innerHTML = originalHTML;
      }}
    }}
    document.getElementById("backtest-run-btn").addEventListener("click", () => {{ runBacktest(); closeFlyouts(); }});

    function renderLeaderboard(data) {{
      const container = document.getElementById("strategies-leaderboard");
      const rows = data.results.map((r, i) => {{
        const m = r.metrics;
        const isBest = r.strategy === data.best_strategy;
        return "<tr" + (isBest ? ' style="background:rgba(41,98,255,0.12);"' : "") + ">" +
          "<td>" + (i + 1) + "</td>" +
          "<td>" + r.display_name + (isBest ? " " + ICON_TROPHY : "") + "</td>" +
          "<td>" + metricPct(m.total_return) + "</td>" +
          "<td>" + metricNum(m.sharpe_ratio) + "</td>" +
          "<td>" + metricPct(m.win_rate) + "</td>" +
          "<td>" + m.num_trades + "</td>" +
        "</tr>";
      }}).join("");
      let benchmarkRow = "";
      if (data.benchmark) {{
        const bm = data.benchmark.metrics;
        benchmarkRow = '<tr style="color:#7d8296; border-top:1px solid #2f3444;">' +
          "<td>--</td><td>Buy &amp; Hold (benchmark)</td>" +
          "<td>" + metricPct(bm.total_return) + "</td>" +
          "<td>" + metricNum(bm.sharpe_ratio) + "</td>" +
          "<td>" + metricPct(bm.win_rate) + "</td>" +
          "<td>" + bm.num_trades + "</td>" +
        "</tr>";
      }}
      let failedNote = "";
      if (data.failed && data.failed.length) {{
        failedNote = '<p style="font-size:11px; color:#7d8296; margin-top:8px;">' +
          data.failed.length + " strategy(ies) failed to run on this data range.</p>";
      }}
      container.innerHTML =
        '<table style="width:100%; font-size:12px;">' +
        "<tr><th>#</th><th>Strategy</th><th>Return</th><th>Sharpe</th><th>Win rate</th><th>Trades</th></tr>" +
        rows + benchmarkRow +
        "</table>" + failedNote;
    }}

    async function runComparison() {{
      const container = document.getElementById("strategies-leaderboard");
      if (!currentSymbol) {{
        container.innerHTML = '<p style="color:#7d8296; font-size:13px;">Load a symbol first.</p>';
        return;
      }}
      container.innerHTML = '<p style="color:#7d8296; font-size:13px;">Running all strategies on ' + currentSymbol + " (" + currentInterval + ")...</p>";
      try {{
        const resp = await fetch("/api/compare/" + encodeURIComponent(currentSymbol), {{
          method: "POST", headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{interval: currentInterval}}),
        }});
        const body = await resp.json();
        if (!resp.ok) {{
          container.innerHTML = '<p style="color:#ef5350; font-size:13px;">' + (body.error || "Comparison failed") + "</p>";
          return;
        }}
        renderLeaderboard(body);
      }} catch (err) {{
        container.innerHTML = '<p style="color:#ef5350; font-size:13px;">Request failed: ' + err + "</p>";
      }}
    }}
    document.getElementById("backtest-compare-btn").addEventListener("click", () => {{
      closeFlyouts();
      setStrategiesPanel(true);
      runComparison();
    }});

    function applyLayout(layout) {{
      restoringLayout = true;
      (layout.drawings || []).forEach((d) => {{
        if (d.type === "hline") addHLineDrawing(d.params.price, true);
        else if (d.type === "trendline") addTrendlineDrawing(d.params.p1, d.params.p2, true);
        else if (d.type === "channel") addChannelDrawing(d.params.a, d.params.b, d.params.p3, true);
        else if (d.type === "ray") addRayDrawing(d.params.p1, d.params.p2, true);
        else if (d.type === "vline") addVLineDrawing(d.params.time, true);
        else if (d.type === "rect") addRectDrawing(d.params.p1, d.params.p2, true);
        else if (d.type === "fib") addFibDrawing(d.params.p1, d.params.p2, true);
        else if (d.type === "fibext") addFibExtDrawing(d.params.p1, d.params.p2, d.params.p3, true);
        else if (d.type === "fibfan") addFibFanDrawing(d.params.p1, d.params.p2, true);
        else if (d.type === "pitchfork") addPitchforkDrawing(d.params.p1, d.params.p2, d.params.p3, true);
        else if (d.type === "text") addTextDrawing({{time: d.params.time, price: d.params.price}}, d.params.text, true);
      }});
      (layout.indicators || []).forEach((ind) => addIndicator(ind.type, ind.params, true));
      (layout.panes || []).forEach((type) => createPane(type, true));
      restoringLayout = false;
    }}

    // The layout embedded in `entry` (SYMBOL_DATA) is only as fresh as the
    // last time this symbol/interval was fetched -- edits made after that
    // (a new drawing, a removed indicator) only land in chart_layouts.json,
    // not in the already-embedded snapshot. So always ask the server for
    // the live layout first; the embedded copy is just the fallback for
    // when this HTML is opened as a static file with no server behind it.
    async function restoreLayout(symbol, interval, embeddedFallback) {{
      let layout = embeddedFallback || {{drawings: [], indicators: []}};
      try {{
        const resp = await fetch("/api/layout/" + encodeURIComponent(symbol) + "?interval=" + encodeURIComponent(interval));
        if (resp.ok) layout = await resp.json();
      }} catch (err) {{ /* no server -- fall back to the embedded snapshot */ }}
      if (currentSymbol !== symbol || currentInterval !== interval) return;  // user moved on before this resolved
      applyLayout(layout);
    }}

    function updateMarkers() {{
      if (!currentSymbol) return;
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      indexBarsByTime(entry);
      clearOverlayMarkers();

      enabledPatterns.forEach((name) => {{
        const abbr = PATTERN_ABBR[name] || name.slice(0, 3).toUpperCase();
        if (entry.patterns && entry.patterns[name]) {{
          entry.patterns[name].forEach((mk) => {{
            const bar = entry._barsByTime[mk.time];
            if (!bar) return;
            const pointDown = mk.position === "aboveBar";
            const price = pointDown ? bar.high : bar.low;
            addOverlayMarker(houseMarkerEl(abbr, mk.color, pointDown), mk.time, price, pointDown ? "above" : "below");
          }});
        }} else if (entry.chart_patterns && entry.chart_patterns[name]) {{
          const color = CHART_PATTERN_COLORS[name] || "#7d8296";
          entry.chart_patterns[name].forEach((occ) => {{
            const arrow = occ.direction === "bullish" ? " ▲" : occ.direction === "bearish" ? " ▼" : "";
            const start = occ.vertices[0];
            addOverlayMarker(chartLabelEl(abbr, color), start.time, start.value, "mid");
            addOverlayMarker(chartLabelEl(abbr + arrow, color), occ.breakout.time, occ.breakout.value, "mid");
          }});
        }}
      }});

      positionOverlayMarkers();
      renderActivePatternTags();
      syncReliabilityPanelHighlights();
      updateRecommendation();
    }}

    function syncReliabilityPanelHighlights() {{
      document.querySelectorAll("#stats-table .pattern-glyph-cell").forEach((el) => {{
        el.classList.toggle("pattern-active", enabledPatterns.has(el.dataset.patternName));
      }});
    }}

    // Zone bands (0-100 gauge-space) per metric, matching the same
    // thresholds compute_metrics_snapshot() uses for its text labels.
    // RSI/Stochastic/MACD are directional (green=bullish-tilt, red=
    // bearish-tilt); ADX/ATR%/Volume measure strength/volatility/turnout,
    // not direction, so they get a neutral blue/amber "intensity" scheme
    // instead of green/red (a strong downtrend scores ADX just as high as
    // a strong uptrend -- coloring that red or green would misrepresent it).
    const GAUGE_ZONES = {{
      "RSI": [{{from: 0, to: 30, color: "#26a69a"}}, {{from: 30, to: 70, color: "#3a3f4d"}}, {{from: 70, to: 100, color: "#ef5350"}}],
      "Stochastic": [{{from: 0, to: 20, color: "#26a69a"}}, {{from: 20, to: 80, color: "#3a3f4d"}}, {{from: 80, to: 100, color: "#ef5350"}}],
      "MFI": [{{from: 0, to: 20, color: "#26a69a"}}, {{from: 20, to: 80, color: "#3a3f4d"}}, {{from: 80, to: 100, color: "#ef5350"}}],
      "MACD": [{{from: 0, to: 50, color: "#ef5350"}}, {{from: 50, to: 100, color: "#26a69a"}}],
      "VWAP Dev": [{{from: 0, to: 50, color: "#ef5350"}}, {{from: 50, to: 100, color: "#26a69a"}}],
      "ADX": [{{from: 0, to: 100, color: "#42a5f5"}}],
      "ATR%": [{{from: 0, to: 50, color: "#42a5f5"}}, {{from: 50, to: 100, color: "#ffa726"}}],
      "Volume": [{{from: 0, to: 50, color: "#3a3f4d"}}, {{from: 50, to: 100, color: "#7e57c2"}}],
    }};

    function buildGaugeSVG(pct, zones, scale) {{
      // JS's Math.max/min (unlike Python's) propagate NaN rather than
      // ignoring it, so any non-finite pct -- from a transient render
      // reading a symbol/interval entry mid-switch, or any future data
      // gap -- would otherwise silently corrupt every coordinate below
      // into NaN. Neutral (50) is the same fallback compute_metrics_snapshot
      // already uses server-side when a reading itself is unavailable.
      pct = Number.isFinite(pct) ? pct : 50;
      scale = scale || 1;
      const w = 100 * scale, h = 60 * scale;
      const cx = 50 * scale, cy = 52 * scale, r = 40 * scale;
      const sw = 8 * scale;
      const toRad = (p) => Math.PI - (Math.max(0, Math.min(100, p)) / 100) * Math.PI;
      const pt = (p, radius) => {{
        const a = toRad(p);
        return [cx + radius * Math.cos(a), cy - radius * Math.sin(a)];
      }};
      // Every zone spans at most a full semicircle (0..100 -> 180deg), so
      // the arc should always sweep the short way across the top -- large-
      // arc-flag=1 here would send mid-range zones (e.g. Stochastic's
      // 20..80 band) the long way around through the bottom, off the
      // visible viewBox entirely.
      let arcs = "";
      zones.forEach((z) => {{
        const [x1, y1] = pt(z.from, r);
        const [x2, y2] = pt(z.to, r);
        arcs += '<path d="M ' + x1 + ' ' + y1 + ' A ' + r + ' ' + r + ' 0 0 1 ' + x2 + ' ' + y2 +
          '" fill="none" stroke="' + z.color + '" stroke-width="' + sw + '" stroke-linecap="round"/>';
      }});
      // Decorative outer bezel ring + tick marks at zone boundaries, for a
      // more "instrument dial" look.
      const [ox1, oy1] = pt(0, r + 6 * scale);
      const [ox2, oy2] = pt(100, r + 6 * scale);
      const ring = '<path d="M ' + ox1 + ' ' + oy1 + ' A ' + (r + 6 * scale) + ' ' + (r + 6 * scale) +
        ' 0 0 1 ' + ox2 + ' ' + oy2 + '" fill="none" stroke="rgba(255,255,255,0.10)" stroke-width="' + (1.25 * scale) + '"/>';
      const boundaries = new Set();
      zones.forEach((z) => {{ boundaries.add(z.from); boundaries.add(z.to); }});
      let ticks = "";
      boundaries.forEach((b) => {{
        if (b <= 0 || b >= 100) return;
        const [tx1, ty1] = pt(b, r - sw / 2 - 1);
        const [tx2, ty2] = pt(b, r + sw / 2 + 1);
        ticks += '<line x1="' + tx1 + '" y1="' + ty1 + '" x2="' + tx2 + '" y2="' + ty2 +
          '" stroke="#0f1117" stroke-width="' + (1.5 * scale) + '"/>';
      }});
      const [nx, ny] = pt(pct, r - sw);
      const needle = '<line x1="' + cx + '" y1="' + cy + '" x2="' + nx + '" y2="' + ny +
        '" stroke="#e6e8ee" stroke-width="' + (2.5 * scale) + '" stroke-linecap="round"/><circle cx="' + cx + '" cy="' + cy + '" r="' + (3.5 * scale) + '" fill="#e6e8ee"/>';
      return '<svg width="' + (w + 12 * scale) + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">' + ring + arcs + ticks + needle + '</svg>';
    }}

    // Hero layout: one metric large and centered, the rest arranged around
    // it in a 3x3 grid (see .osc-grid) -- Volume front-and-center per user
    // preference, everything else placed by feel (RSI/MACD up top since
    // they're the most-watched momentum pair, Stochastic/ADX below).
    const OSC_LAYOUT = {{
      "Volume": {{area: "center", scale: 1.6}},
      "RSI": {{area: "tl", scale: 1}},
      "ATR%": {{area: "top", scale: 1}},
      "MACD": {{area: "tr", scale: 1}},
      "MFI": {{area: "ml", scale: 1}},
      "VWAP Dev": {{area: "mr", scale: 1}},
      "Stochastic": {{area: "bl", scale: 1}},
      "ADX": {{area: "br", scale: 1}},
    }};

    // Small context labels under each gauge's low/high ends.
    const GAUGE_END_LABELS = {{
      "RSI": ["Oversold", "Overbought"], "Stochastic": ["Oversold", "Overbought"],
      "MFI": ["Oversold", "Overbought"], "VWAP Dev": ["Below VWAP", "Above VWAP"],
      "MACD": ["Bearish", "Bullish"], "ADX": ["Ranging", "Trending"],
      "ATR%": ["Low", "Elevated"], "Volume": ["Below avg", "Above avg"],
    }};

    function renderMetricsDashboard(entry) {{
      const container = document.getElementById("metrics-dashboard");
      container.innerHTML = "";
      (entry.metrics || []).forEach((m) => {{
        const layout = OSC_LAYOUT[m.name] || {{area: "bottom", scale: 1}};
        const isHero = layout.area === "center";
        const card = document.createElement("div");
        card.className = "osc-card" + (isHero ? " osc-hero" : "");
        card.style.gridArea = layout.area;
        const zones = GAUGE_ZONES[m.name] || [{{from: 0, to: 100, color: "#3a3f4d"}}];
        const ends = GAUGE_END_LABELS[m.name] || ["", ""];
        card.title = m.detail;
        card.innerHTML = "<div style='font-size:" + (isHero ? 13 : 11) + "px; color:#7d8296;'>" + m.name + "</div>" +
          buildGaugeSVG(m.gauge_pct, zones, layout.scale) +
          "<div style='display:flex; justify-content:space-between; font-size:" + (isHero ? 10 : 9) +
          "px; color:#5a5f70; margin-top:-4px; padding:0 " + (isHero ? 10 : 4) + "px;'>" +
          "<span>" + ends[0] + "</span><span>" + ends[1] + "</span></div>" +
          "<div style='font-size:" + (isHero ? 19 : 15) + "px; font-weight:bold; margin-top:2px;'>" + m.value + "</div>" +
          "<div style='font-size:" + (isHero ? 13 : 11) + "px; display:flex; align-items:center; justify-content:center; gap:4px;'>" +
          directionIcon(m.label) + "<span>" + m.label + "</span></div>";
        container.appendChild(card);
      }});
    }}

    function findPatternStats(entry, name) {{
      const inCandle = (entry.pattern_stats || []).find(s => s.name === name);
      if (inCandle) return inCandle;
      return (entry.chart_pattern_stats || []).find(s => s.name === name);
    }}

    function updateRecommendation() {{
      const box = document.getElementById("recommendation");
      if (!currentSymbol || enabledPatterns.size === 0) {{
        box.textContent = "No patterns selected -- showing price action only.";
        return;
      }}
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      let score = 0;
      let bullish = [], bearish = [];

      enabledPatterns.forEach((name) => {{
        const stats = findPatternStats(entry, name);
        if (!stats || stats.direction === "neutral" || stats.occurrences === 0) return;
        const winRate = stats.win_rate === null || stats.win_rate === undefined ? 0.5 : stats.win_rate;
        const confidence = (stats.avg_confidence === null || stats.avg_confidence === undefined ? 50 : stats.avg_confidence) / 100;
        const sampleWeight = Math.min(1, stats.occurrences / MIN_OCCURRENCES_FOR_WIN_RATE);
        const weight = winRate * confidence * sampleWeight;
        const sign = stats.direction === "bullish" ? 1 : -1;
        score += sign * weight;
        (sign > 0 ? bullish : bearish).push(name + " (" + Math.round(winRate * 100) + "%)");
      }});

      let verdict, verdictIcon;
      if (Math.abs(score) < 0.05) {{
        verdict = "Mixed / no clear bias"; verdictIcon = "";
      }} else if (score > 0) {{
        verdict = "Bullish bias"; verdictIcon = "<span style='color:#26a69a; display:inline-flex;'>" + ICON_TRENDING_UP + "</span>";
      }} else {{
        verdict = "Bearish bias"; verdictIcon = "<span style='color:#ef5350; display:inline-flex;'>" + ICON_TRENDING_DOWN + "</span>";
      }}

      let breakdown = [];
      if (bullish.length) breakdown.push(bullish.length + " bullish: " + bullish.join(", "));
      if (bearish.length) breakdown.push(bearish.length + " bearish: " + bearish.join(", "));

      box.innerHTML = "<strong style='display:inline-flex; align-items:center; gap:6px;'>" + verdictIcon + verdict + "</strong> (score " + score.toFixed(2) + ")" +
        (breakdown.length ? "<br><span style='font-size:13px; color:#7d8296;'>" + breakdown.join(" | ") + "</span>" : "") +
        "<br><span style='font-size:12px; color:#7d8296;'>Weighted by each pattern type's empirical win rate, confidence, and sample size -- not a guarantee.</span>";
    }}

    function togglePattern(name) {{
      if (!currentSymbol) {{
        patternHint("Load a symbol first.");
        return;
      }}
      const entry = SYMBOL_DATA[entryKey(currentSymbol, currentInterval)];
      const isChartPattern = entry.chart_patterns && entry.chart_patterns[name];
      const isCandlePattern = entry.patterns && entry.patterns[name];

      if (enabledPatterns.has(name)) {{
        enabledPatterns.delete(name);
        if (isChartPattern) removeChartPatternLines(name);
        patternHint("Hid pattern: " + name);
      }} else if (isChartPattern || isCandlePattern) {{
        enabledPatterns.add(name);
        if (isChartPattern) renderChartPatternLines(name);
        patternHint("Showing pattern: " + name);
      }} else {{
        patternHint("No " + name + " occurrences found for " + currentSymbol);
        return;
      }}
      updateMarkers();
    }}

    function entryKey(symbol, interval) {{ return symbol + "@" + interval; }}

    function getDistinctSymbols() {{
      const set = new Set();
      Object.values(SYMBOL_DATA).forEach((e) => set.add(e.symbol));
      return Array.from(set);
    }}

    function updateIntervalButtons() {{
      document.querySelectorAll("#interval-toolbar button[data-interval]").forEach((b) => {{
        b.classList.toggle("tool-btn-active", b.dataset.interval === currentInterval);
      }});
    }}

    function loadEntry(symbol, interval) {{
      const entry = SYMBOL_DATA[entryKey(symbol, interval)];
      if (!entry) return false;
      currentSymbol = symbol;
      currentInterval = interval;
      priceSeries.setData(entry.bars);
      volumeSeries.setData(entry.volume);
      mainLinkEntry.dataByTime = seriesToDataByTime(entry.bars.map((b) => ({{time: b.time, value: b.close}})));
      Object.keys(chartPatternSeriesMap).forEach(name => removeChartPatternLines(name));
      enabledPatterns = new Set();
      clearOverlayMarkers();
      renderActivePatternTags();
      clearDrawings();
      clearIndicators();
      clearAllPanes();
      clearBacktest();
      setActiveTool("cursor");
      chart.timeScale().fitContent();
      restoreLayout(symbol, interval, entry.layout);
      renderMetricsDashboard(entry);
      updateRecommendation();
      document.getElementById("stats-table").innerHTML = entry.stats_table_html;
      document.getElementById("current-symbol-label").textContent =
        "Currently viewing: " + symbol + " (" + entry.interval + ")";
      updateIntervalButtons();
      symbolHint("");
      patternHint("");
      renderWatchlist();
      return true;
    }}

    function findSymbolMatch(q) {{
      const syms = getDistinctSymbols();
      return syms.find(s => s.toLowerCase() === q) || syms.find(s => s.toLowerCase().startsWith(q));
    }}

    function addToSymbolDatalist(symbol) {{
      const opt = document.createElement("option");
      opt.value = symbol;
      document.getElementById("symbol-search-options").appendChild(opt);
    }}

    async function loadOrFetch(symbol, interval) {{
      if (loadEntry(symbol, interval)) return;
      symbolHint("Looking up " + symbol + " (" + interval + ") ...");
      try {{
        const resp = await fetch("/api/symbol/" + encodeURIComponent(symbol) + "?interval=" + encodeURIComponent(interval));
        const body = await resp.json();
        if (!resp.ok) {{ symbolHint(body.error || ("No data found for " + symbol)); return; }}
        const wasKnown = getDistinctSymbols().includes(symbol);
        SYMBOL_DATA[entryKey(symbol, interval)] = body;
        if (!wasKnown) addToSymbolDatalist(symbol);
        loadEntry(symbol, interval);
      }} catch (err) {{
        symbolHint("Live lookup failed (is the server running via 'python main.py --serve'?): " + err);
      }}
    }}

    function handleSymbolInput(rawQuery) {{
      // Fires on every keystroke -- switching to an already-loaded
      // symbol@interval combo is idempotent/safe to repeat here; a
      // brand-new lookup only happens on Enter/Search click (symbolSearch).
      const q = rawQuery.trim().toLowerCase();
      if (!q) {{ symbolHint(""); return; }}
      const symbolMatch = findSymbolMatch(q);
      if (symbolMatch) {{
        if (symbolMatch !== currentSymbol) loadEntry(symbolMatch, currentInterval);
      }} else {{
        symbolHint("No loaded symbol matches \\"" + rawQuery + "\\" -- press Enter or click Search to look it up live.");
      }}
    }}

    async function symbolSearch(rawQuery) {{
      const trimmed = rawQuery.trim();
      if (!trimmed) return;
      const symbolMatch = findSymbolMatch(trimmed.toLowerCase());
      const symbol = symbolMatch || trimmed.toUpperCase();
      await loadOrFetch(symbol, currentInterval);
    }}

    document.getElementById("interval-toolbar").addEventListener("click", (e) => {{
      const btn = e.target.closest("button[data-interval]");
      if (!btn || !currentSymbol || btn.dataset.interval === currentInterval) return;
      loadOrFetch(currentSymbol, btn.dataset.interval);
    }});

    function handlePatternInput(rawQuery) {{
      // Passive only -- toggling isn't idempotent, so it never fires on
      // keystrokes, only on Enter/Toggle click (patternSearch below).
      const q = rawQuery.trim().toLowerCase();
      if (!q) {{ patternHint(""); return; }}
      const patternMatch = PATTERN_NAMES.find(p => p.includes(q));
      patternHint(patternMatch
        ? "Press Enter or click Toggle to show/hide: " + patternMatch
        : "No pattern matches \\"" + rawQuery + "\\"");
    }}

    function patternSearch(rawQuery) {{
      const q = rawQuery.trim().toLowerCase();
      if (!q) return;
      const patternMatch = PATTERN_NAMES.find(p => p.includes(q));
      if (patternMatch) {{
        togglePattern(patternMatch);
      }} else {{
        patternHint("No pattern matches \\"" + rawQuery + "\\"");
      }}
    }}

    document.getElementById("symbol-search-box").addEventListener("input", (e) => handleSymbolInput(e.target.value));
    document.getElementById("symbol-search-box").addEventListener("keydown", (e) => {{
      if (e.key === "Enter") symbolSearch(e.target.value);
    }});
    document.getElementById("symbol-search-button").addEventListener("click", () => {{
      symbolSearch(document.getElementById("symbol-search-box").value);
    }});

    document.getElementById("pattern-search-box").addEventListener("input", (e) => handlePatternInput(e.target.value));
    document.getElementById("pattern-search-box").addEventListener("keydown", (e) => {{
      if (e.key === "Enter") patternSearch(e.target.value);
    }});
    document.getElementById("pattern-search-button").addEventListener("click", () => {{
      patternSearch(document.getElementById("pattern-search-box").value);
    }});

    renderWatchlist();
    populateBacktestStrategySelect();
    if ({json.dumps(default_key)}) {{
      const defaultEntry = SYMBOL_DATA[{json.dumps(default_key)}];
      loadEntry(defaultEntry.symbol, defaultEntry.interval);
    }}
  </script>
</body>
</html>"""


def _pattern_stats_to_dicts(pattern_stats: List[PatternStats]) -> List[dict]:
    """Structured (not pre-rendered HTML) form of PatternStats, for the
    dashboard's live recommendation aggregate to compute over in JS. NaN
    serializes as JSON null (pd.isna -> None) rather than the string
    'NaN', which plain json.dumps would otherwise choke on."""
    return [
        {
            "name": s.name,
            "direction": s.direction,
            "occurrences": s.occurrences,
            "win_rate": None if pd.isna(s.win_rate) else s.win_rate,
            "avg_confidence": None if pd.isna(s.avg_confidence) else s.avg_confidence,
        }
        for s in pattern_stats
    ]


def compute_symbol_entry(symbol: str, interval: str, bars: pd.DataFrame, forward_bars: int = 10) -> dict:
    """Builds one symbol's dashboard-store entry. Shared by the CLI path
    (update_patterns_dashboard) and the live-lookup Flask endpoint
    (stockx.server) so both produce identical entries. Computes both
    candlestick and chart pattern stats internally (rather than requiring
    callers to precompute them) so both call sites stay simple."""
    payload = _build_chart_payload(bars)
    all_chart_patterns = find_all_chart_patterns(bars)  # computed once, reused below

    pattern_stats = compute_pattern_stats(bars, forward_bars=forward_bars)
    chart_pattern_stats = compute_chart_pattern_stats(bars, all_chart_patterns, forward_bars=forward_bars)
    metrics = compute_metrics_snapshot(bars)

    return {
        "symbol": symbol,
        "interval": interval,
        "generated_at": datetime.now().isoformat(),
        "bars": payload["bars"],
        "volume": payload["volume"],
        "patterns": payload["patterns"],
        "chart_patterns": _build_chart_pattern_payload(all_chart_patterns),
        "stats_table_html": _stats_table_html(pattern_stats, chart_pattern_stats, forward_bars),
        "pattern_stats": _pattern_stats_to_dicts(pattern_stats),
        "chart_pattern_stats": _pattern_stats_to_dicts(chart_pattern_stats),
        "layout": get_layout(symbol, interval),
        "metrics": [
            {"name": m.name, "value": m.value, "label": m.label, "detail": m.detail, "gauge_pct": m.gauge_pct}
            for m in metrics
        ],
    }


def update_patterns_dashboard(symbol: str, interval: str, bars: pd.DataFrame, forward_bars: int = 10) -> tuple:
    """Adds/refreshes `symbol`'s entry (at this specific interval) in the
    persistent dashboard and rewrites reports/patterns_dashboard.html from
    the full accumulated store. Entries are keyed by "SYMBOL@interval" so a
    symbol can have multiple timeframes cached side by side without one
    overwriting another. Returns (dashboard_html_path, total_entry_count)."""
    symbol = symbol.upper()
    data = _load_dashboard_data()
    data[f"{symbol}@{interval}"] = compute_symbol_entry(symbol, interval, bars, forward_bars)

    _save_dashboard_data(data)
    DASHBOARD_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HTML_PATH.write_text(_render_dashboard_html(data, load_watchlist()))
    return DASHBOARD_HTML_PATH, len(data)
