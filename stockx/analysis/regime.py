import numpy as np
import pandas as pd

from stockx.strategies.indicators import adx, atr_pct


def classify_regime(bars: pd.DataFrame, adx_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """Tag each bar with a market regime, aligned to bars.index.

    Buckets are median splits computed on this symbol's own historical
    distribution (not fixed absolute thresholds like 'ADX > 25'), so the
    same logic is meaningful for any symbol/asset class regardless of its
    typical volatility or trendiness.
    """
    trend_strength = adx(bars, adx_period)
    volatility_pct = atr_pct(bars, atr_period)

    trend_median = trend_strength.median()
    vol_median = volatility_pct.median()

    trend_regime = np.where(trend_strength >= trend_median, "trending", "ranging")
    vol_regime = np.where(volatility_pct >= vol_median, "high_vol", "low_vol")

    return pd.DataFrame(
        {
            "trend_strength": trend_strength,
            "volatility_pct": volatility_pct,
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
        },
        index=bars.index,
    )
