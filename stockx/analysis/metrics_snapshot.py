import math
from dataclasses import dataclass
from typing import List

import pandas as pd

from stockx.strategies.indicators import adx, atr_pct, macd, money_flow_index, rsi, session_vwap, stochastic


@dataclass
class MetricReading:
    name: str
    value: str
    label: str
    detail: str
    gauge_pct: float  # 0-100 needle position for a speedometer-style gauge


def _safe_last(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    value = series.iloc[-1]
    return float(value) if not pd.isna(value) else float("nan")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_metrics_snapshot(bars: pd.DataFrame) -> List[MetricReading]:
    """Current-bar readings for a fixed set of oscillators/metrics, each
    labeled by simple documented thresholds -- auditable fixed rules, same
    spirit as the candlestick/chart pattern thresholds elsewhere in this
    project, not adaptive fitting. Each reading also carries a gauge_pct
    (0-100) for a speedometer-style dashboard display: RSI/Stochastic/ADX
    are already 0-100 bounded so gauge_pct is the raw value; MACD/ATR%/
    Volume have no fixed scale, so gauge_pct is centered at 50 (= neutral/
    at its own baseline) and diverges toward 0 or 100 based on how far the
    current reading sits from that baseline, capped at a fixed swing so one
    extreme bar can't peg the needle."""
    readings: List[MetricReading] = []

    rsi_val = _safe_last(rsi(bars["close"]))
    if math.isnan(rsi_val):
        rsi_label, rsi_gauge = "Neutral", 50.0
    else:
        rsi_gauge = _clamp(rsi_val)
        if rsi_val > 70:
            rsi_label = "Overbought"
        elif rsi_val < 30:
            rsi_label = "Oversold"
        else:
            rsi_label = "Neutral"
    readings.append(MetricReading(
        name="RSI", value=f"{rsi_val:.1f}" if not math.isnan(rsi_val) else "N/A",
        label=rsi_label, detail=f"RSI(14) {rsi_val:.1f} -- {rsi_label.lower()} (overbought >70, oversold <30)",
        gauge_pct=rsi_gauge,
    ))

    macd_line, signal_line, hist = macd(bars["close"])
    m, s, h = _safe_last(macd_line), _safe_last(signal_line), _safe_last(hist)
    close_price = _safe_last(bars["close"])
    if math.isnan(h):
        macd_label, macd_gauge = "Neutral", 50.0
    else:
        macd_label = "Bullish" if h > 0 else "Bearish" if h < 0 else "Neutral"
        # Histogram has no fixed scale (depends on the symbol's price level),
        # so express it as a fraction of price: a swing of +-2% of price
        # pegs the needle fully bullish/bearish.
        swing = 0.0 if not close_price or math.isnan(close_price) else h / (0.02 * close_price)
        macd_gauge = _clamp(50 + swing * 50)
    readings.append(MetricReading(
        name="MACD", value=f"{m:.2f} / {s:.2f}" if not math.isnan(m) else "N/A",
        label=macd_label, detail=f"MACD {m:.2f}, Signal {s:.2f}, Hist {h:+.2f} -- {macd_label.lower()}",
        gauge_pct=macd_gauge,
    ))

    k_series, d_series = stochastic(bars)
    k_val, d_val = _safe_last(k_series), _safe_last(d_series)
    if math.isnan(k_val):
        stoch_label, stoch_gauge = "Neutral", 50.0
    else:
        stoch_gauge = _clamp(k_val)
        if k_val > 80:
            stoch_label = "Overbought"
        elif k_val < 20:
            stoch_label = "Oversold"
        else:
            stoch_label = "Neutral"
    readings.append(MetricReading(
        name="Stochastic", value=f"{k_val:.1f} / {d_val:.1f}" if not math.isnan(k_val) else "N/A",
        label=stoch_label, detail=f"%K {k_val:.1f}, %D {d_val:.1f} -- {stoch_label.lower()} (overbought >80, oversold <20)",
        gauge_pct=stoch_gauge,
    ))

    adx_val = _safe_last(adx(bars))
    if math.isnan(adx_val):
        adx_label, adx_gauge = "Transitional", 30.0
    else:
        adx_gauge = _clamp(adx_val)
        if adx_val > 25:
            adx_label = "Trending"
        elif adx_val < 20:
            adx_label = "Ranging"
        else:
            adx_label = "Transitional"
    readings.append(MetricReading(
        name="ADX", value=f"{adx_val:.1f}" if not math.isnan(adx_val) else "N/A",
        label=adx_label, detail=f"ADX(14) {adx_val:.1f} -- {adx_label.lower()} (trend strength only, no direction)",
        gauge_pct=adx_gauge,
    ))

    atr_series = atr_pct(bars).dropna()
    if len(atr_series):
        current_atr = float(atr_series.iloc[-1])
        median_atr = float(atr_series.median())
        ratio = current_atr / median_atr if median_atr > 0 else 1.0
        if ratio > 1.5:
            atr_label = "Elevated"
        elif ratio < 0.67:
            atr_label = "Low"
        else:
            atr_label = "Normal"
        atr_value = f"{current_atr * 100:.2f}%"
        atr_gauge = _clamp(50 + (ratio - 1) * 50)
    else:
        current_atr, atr_label, atr_value, atr_gauge = float("nan"), "Normal", "N/A", 50.0
    readings.append(MetricReading(
        name="ATR%", value=atr_value, label=atr_label,
        detail=f"ATR {atr_value} of price -- {atr_label.lower()} vs this symbol's own trailing median",
        gauge_pct=atr_gauge,
    ))

    volume = bars["volume"]
    if len(volume):
        current_vol = float(volume.iloc[-1])
        avg_vol = float(volume.rolling(20, min_periods=1).mean().iloc[-1])
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if ratio > 1.3:
            vol_label = "Above average"
        elif ratio < 0.7:
            vol_label = "Below average"
        else:
            vol_label = "Average"
        vol_value = f"{current_vol:,.0f} ({ratio:.2f}x avg)"
        vol_gauge = _clamp(50 + (ratio - 1) * 50)
    else:
        vol_label, vol_value, ratio, vol_gauge = "Average", "N/A", 1.0, 50.0
    readings.append(MetricReading(
        name="Volume", value=vol_value, label=vol_label,
        detail=f"Latest bar volume {vol_value} vs trailing 20-bar average",
        gauge_pct=vol_gauge,
    ))

    mfi_val = _safe_last(money_flow_index(bars))
    if math.isnan(mfi_val):
        mfi_label, mfi_gauge = "Neutral", 50.0
    else:
        mfi_gauge = _clamp(mfi_val)
        if mfi_val > 80:
            mfi_label = "Overbought"
        elif mfi_val < 20:
            mfi_label = "Oversold"
        else:
            mfi_label = "Neutral"
    readings.append(MetricReading(
        name="MFI", value=f"{mfi_val:.1f}" if not math.isnan(mfi_val) else "N/A",
        label=mfi_label, detail=f"MFI(14) {mfi_val:.1f} -- {mfi_label.lower()} (volume-weighted RSI; overbought >80, oversold <20)",
        gauge_pct=mfi_gauge,
    ))

    vwap_series = session_vwap(bars).dropna()
    if len(vwap_series) and not math.isnan(close_price):
        current_vwap = float(vwap_series.iloc[-1])
        vwap_dev = (close_price - current_vwap) / current_vwap if current_vwap > 0 else 0.0
        if vwap_dev > 0.003:
            vwap_label = "Above VWAP"
        elif vwap_dev < -0.003:
            vwap_label = "Below VWAP"
        else:
            vwap_label = "At VWAP"
        vwap_value = f"{vwap_dev:+.2%}"
        # +-1% of price covers the needle's full swing -- VWAP deviations
        # are usually much tighter than MACD's price-relative swing.
        vwap_gauge = _clamp(50 + (vwap_dev / 0.01) * 50)
    else:
        vwap_label, vwap_value, vwap_gauge = "At VWAP", "N/A", 50.0
    readings.append(MetricReading(
        name="VWAP Dev", value=vwap_value, label=vwap_label,
        detail=f"Price vs session VWAP: {vwap_value} -- {vwap_label.lower()} (resets each trading day)",
        gauge_pct=vwap_gauge,
    ))

    return readings
