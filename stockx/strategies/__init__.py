from typing import Dict, List, Optional, Type

from stockx.strategies.base import STRATEGY_REGISTRY, Strategy

# Import every strategy module purely for its @register side effect.
from stockx.strategies import (  # noqa: E402,F401
    orb,
    vwap_reversion,
    ma_crossover,
    rsi_reversion,
    volume_momentum,
    candlestick_reversal,
    range_breakout,
    chart_pattern_breakout,
    awesome_oscillator,
    macd_crossover,
    parabolic_sar,
    dual_thrust,
    heikin_ashi_trend,
    bollinger_reversion,
)


def get_registered_strategies(overrides: Optional[Dict[str, dict]] = None) -> List[Strategy]:
    overrides = overrides or {}
    return [cls(**overrides.get(name, {})) for name, cls in STRATEGY_REGISTRY.items()]


def get_strategy_names() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())
