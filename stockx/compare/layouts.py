import json
from typing import Dict, List

from stockx.config import REPORTS_DIR

# Saved chart state (drawings + on-chart indicators) per "SYMBOL@interval",
# same keying scheme as the pattern dashboard store -- one chart's setup is
# independent of the same symbol viewed at a different timeframe.
LAYOUTS_JSON_PATH = REPORTS_DIR / "chart_layouts.json"

_EMPTY_LAYOUT = {"drawings": [], "indicators": [], "panes": []}


def _load_layouts() -> Dict[str, dict]:
    if not LAYOUTS_JSON_PATH.exists():
        return {}
    with open(LAYOUTS_JSON_PATH) as f:
        return json.load(f)


def _save_layouts(layouts: Dict[str, dict]) -> None:
    LAYOUTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAYOUTS_JSON_PATH, "w") as f:
        json.dump(layouts, f)


def get_layout(symbol: str, interval: str) -> dict:
    layouts = _load_layouts()
    return layouts.get(f"{symbol.upper()}@{interval}", dict(_EMPTY_LAYOUT))


def save_layout(
    symbol: str, interval: str, drawings: List[dict], indicators: List[dict], panes: List[str] = None,
) -> None:
    layouts = _load_layouts()
    layouts[f"{symbol.upper()}@{interval}"] = {
        "drawings": drawings, "indicators": indicators, "panes": panes or [],
    }
    _save_layouts(layouts)
