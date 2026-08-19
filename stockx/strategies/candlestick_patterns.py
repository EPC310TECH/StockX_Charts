import numpy as np
import pandas as pd

# Deterministic OHLC-ratio candlestick pattern detectors -- the same style
# as the numeric indicators in indicators.py, just boolean instead of
# numeric. Each function is vectorized over the whole bar series (no
# bar-by-bar loops). NaN produced by shift() during warm-up bars propagates
# through numpy comparisons as False, so early bars correctly report no
# pattern rather than raising or producing spurious True.


def _components(bars: pd.DataFrame, shift: int = 0):
    o = bars["open"].shift(shift)
    h = bars["high"].shift(shift)
    l = bars["low"].shift(shift)
    c = bars["close"].shift(shift)
    body = (c - o).abs()
    rng = h - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    return o, h, l, c, body, rng, upper_wick, lower_wick


def is_doji(bars: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    _, _, _, _, body, rng, _, _ = _components(bars)
    return (body <= body_ratio * rng.replace(0, np.nan)).fillna(False)


def is_hammer_shape(bars: pd.DataFrame) -> pd.Series:
    """Small body near the top of the range, long lower wick, small upper
    wick -- the shared geometry of a hammer/hanging-man."""
    _, _, _, _, body, rng, upper_wick, lower_wick = _components(bars)
    rng_safe = rng.replace(0, np.nan)
    small_body = body <= 0.3 * rng_safe
    long_lower_wick = lower_wick >= 2 * body
    small_upper_wick = upper_wick <= 0.1 * rng_safe
    return (small_body & long_lower_wick & small_upper_wick).fillna(False)


def is_inverted_hammer_shape(bars: pd.DataFrame) -> pd.Series:
    """Small body near the bottom of the range, long upper wick, small
    lower wick -- the shared geometry of a shooting-star/inverted-hammer."""
    _, _, _, _, body, rng, upper_wick, lower_wick = _components(bars)
    rng_safe = rng.replace(0, np.nan)
    small_body = body <= 0.3 * rng_safe
    long_upper_wick = upper_wick >= 2 * body
    small_lower_wick = lower_wick <= 0.1 * rng_safe
    return (small_body & long_upper_wick & small_lower_wick).fillna(False)


def prior_trend(bars: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """'up'/'down'/'flat' price direction over the `lookback` bars ending at
    the PREVIOUS bar (excludes the current candle, so a pattern's own
    candle never influences its trend classification)."""
    prior_close = bars["close"].shift(1)
    baseline_close = bars["close"].shift(1 + lookback)
    change = prior_close - baseline_close
    return pd.Series(
        np.where(change > 0, "up", np.where(change < 0, "down", "flat")),
        index=bars.index,
    )


def is_hammer(bars: pd.DataFrame) -> pd.Series:
    return is_hammer_shape(bars) & (prior_trend(bars) == "down")


def is_hanging_man(bars: pd.DataFrame) -> pd.Series:
    return is_hammer_shape(bars) & (prior_trend(bars) == "up")


def is_shooting_star(bars: pd.DataFrame) -> pd.Series:
    return is_inverted_hammer_shape(bars) & (prior_trend(bars) == "up")


def is_inverted_hammer(bars: pd.DataFrame) -> pd.Series:
    return is_inverted_hammer_shape(bars) & (prior_trend(bars) == "down")


def is_bullish_engulfing(bars: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = bars["open"].shift(1), bars["close"].shift(1)
    prev_bearish = prev_close < prev_open
    curr_bullish = bars["close"] > bars["open"]
    engulfs = (bars["open"] <= prev_close) & (bars["close"] >= prev_open)
    return (prev_bearish & curr_bullish & engulfs).fillna(False)


def is_bearish_engulfing(bars: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = bars["open"].shift(1), bars["close"].shift(1)
    prev_bullish = prev_close > prev_open
    curr_bearish = bars["close"] < bars["open"]
    engulfs = (bars["open"] >= prev_close) & (bars["close"] <= prev_open)
    return (prev_bullish & curr_bearish & engulfs).fillna(False)


def _star_setup(bars: pd.DataFrame):
    """Shared first-two-candle geometry for morning/evening stars: returns
    (c1_bearish, c1_bullish, c1_long, c2_small, c1_midpoint)."""
    c1_open, c1_high, c1_low, c1_close, _, c1_rng, _, _ = _components(bars, shift=2)
    c2_open, _, _, c2_close, c2_body, c2_rng, _, _ = _components(bars, shift=1)

    c1_rng_safe = c1_rng.replace(0, np.nan)
    c1_bearish = c1_close < c1_open
    c1_bullish = c1_close > c1_open
    c1_long = (c1_open - c1_close).abs() >= 0.5 * c1_rng_safe
    c2_small = c2_body <= 0.3 * c2_rng.replace(0, np.nan)
    c1_midpoint = (c1_open + c1_close) / 2
    return c1_bearish, c1_bullish, c1_long, c2_small, c1_midpoint


def is_morning_star(bars: pd.DataFrame) -> pd.Series:
    c1_bearish, _, c1_long, c2_small, c1_midpoint = _star_setup(bars)
    c3_bullish = bars["close"] > bars["open"]
    closes_above_midpoint = bars["close"] > c1_midpoint
    return (c1_bearish & c1_long & c2_small & c3_bullish & closes_above_midpoint).fillna(False)


def is_evening_star(bars: pd.DataFrame) -> pd.Series:
    _, c1_bullish, c1_long, c2_small, c1_midpoint = _star_setup(bars)
    c3_bearish = bars["close"] < bars["open"]
    closes_below_midpoint = bars["close"] < c1_midpoint
    return (c1_bullish & c1_long & c2_small & c3_bearish & closes_below_midpoint).fillna(False)


def morning_star_forming(bars: pd.DataFrame) -> pd.Series:
    """True when the last 2 bars match the first 2 candles of a morning
    star (long bearish candle, then a small-body indecision candle) --
    awaiting a 3rd bullish confirmation candle. The 'actively forming'
    early signal, one bar ahead of is_morning_star confirming."""
    c1_open, _, _, c1_close, _, c1_rng, _, _ = _components(bars, shift=1)
    c2_open, _, _, c2_close, c2_body, c2_rng, _, _ = _components(bars, shift=0)

    c1_bearish = c1_close < c1_open
    c1_long = (c1_open - c1_close).abs() >= 0.5 * c1_rng.replace(0, np.nan)
    c2_small = c2_body <= 0.3 * c2_rng.replace(0, np.nan)
    return (c1_bearish & c1_long & c2_small).fillna(False)


def evening_star_forming(bars: pd.DataFrame) -> pd.Series:
    c1_open, _, _, c1_close, _, c1_rng, _, _ = _components(bars, shift=1)
    c2_open, _, _, c2_close, c2_body, c2_rng, _, _ = _components(bars, shift=0)

    c1_bullish = c1_close > c1_open
    c1_long = (c1_close - c1_open).abs() >= 0.5 * c1_rng.replace(0, np.nan)
    c2_small = c2_body <= 0.3 * c2_rng.replace(0, np.nan)
    return (c1_bullish & c1_long & c2_small).fillna(False)


def is_three_white_soldiers(bars: pd.DataFrame) -> pd.Series:
    o0, h0, l0, c0, _, r0, _, _ = _components(bars, shift=2)
    o1, h1, l1, c1, _, r1, _, _ = _components(bars, shift=1)
    o2, h2, l2, c2, _, r2, _, _ = _components(bars, shift=0)

    bullish = (c0 > o0) & (c1 > o1) & (c2 > o2)
    rising_closes = (c1 > c0) & (c2 > c1)
    opens_within_prior_body = (o1 > o0) & (o1 < c0) & (o2 > o1) & (o2 < c1)
    small_wicks = (
        ((h0 - c0) <= 0.25 * r0.replace(0, np.nan))
        & ((h1 - c1) <= 0.25 * r1.replace(0, np.nan))
        & ((h2 - c2) <= 0.25 * r2.replace(0, np.nan))
    )
    return (bullish & rising_closes & opens_within_prior_body & small_wicks).fillna(False)


def is_three_black_crows(bars: pd.DataFrame) -> pd.Series:
    o0, h0, l0, c0, _, r0, _, _ = _components(bars, shift=2)
    o1, h1, l1, c1, _, r1, _, _ = _components(bars, shift=1)
    o2, h2, l2, c2, _, r2, _, _ = _components(bars, shift=0)

    bearish = (c0 < o0) & (c1 < o1) & (c2 < o2)
    falling_closes = (c1 < c0) & (c2 < c1)
    opens_within_prior_body = (o1 < o0) & (o1 > c0) & (o2 < o1) & (o2 > c1)
    small_wicks = (
        ((c0 - l0) <= 0.25 * r0.replace(0, np.nan))
        & ((c1 - l1) <= 0.25 * r1.replace(0, np.nan))
        & ((c2 - l2) <= 0.25 * r2.replace(0, np.nan))
    )
    return (bearish & falling_closes & opens_within_prior_body & small_wicks).fillna(False)


# --- Geometric confidence scores (0-100) -----------------------------------
# Additive companions to the boolean is_* functions above -- same thresholds,
# turned into a continuous "how strongly is this condition satisfied" ramp
# instead of a hard cutoff, then gated to 0 wherever the boolean pattern is
# False. These do not change any is_* function's behavior.


def _ramp_score(value: pd.Series, low: float, high: float) -> pd.Series:
    """0 at/below `low`, 1 at/above `high`, linear in between."""
    return ((value - low) / (high - low)).clip(lower=0, upper=1)


def doji_confidence(bars: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    _, _, _, _, body, rng, _, _ = _components(bars)
    ratio = body / rng.replace(0, np.nan)
    score = (_ramp_score(-ratio, -body_ratio, 0.0) * 100).fillna(0)
    return score.where(is_doji(bars, body_ratio), 0.0)


def _hammer_shape_confidence(bars: pd.DataFrame) -> pd.Series:
    _, _, _, _, body, rng, upper_wick, lower_wick = _components(bars)
    rng_safe = rng.replace(0, np.nan)
    body_score = _ramp_score(-(body / rng_safe), -0.3, 0.0)
    lower_score = _ramp_score(lower_wick / body.replace(0, np.nan), 2.0, 4.0)
    upper_score = _ramp_score(-(upper_wick / rng_safe), -0.1, 0.0)
    return ((body_score + lower_score + upper_score) / 3 * 100).fillna(0)


def _inverted_hammer_shape_confidence(bars: pd.DataFrame) -> pd.Series:
    _, _, _, _, body, rng, upper_wick, lower_wick = _components(bars)
    rng_safe = rng.replace(0, np.nan)
    body_score = _ramp_score(-(body / rng_safe), -0.3, 0.0)
    upper_score = _ramp_score(upper_wick / body.replace(0, np.nan), 2.0, 4.0)
    lower_score = _ramp_score(-(lower_wick / rng_safe), -0.1, 0.0)
    return ((body_score + upper_score + lower_score) / 3 * 100).fillna(0)


def hammer_confidence(bars: pd.DataFrame) -> pd.Series:
    return _hammer_shape_confidence(bars).where(is_hammer(bars), 0.0)


def hanging_man_confidence(bars: pd.DataFrame) -> pd.Series:
    return _hammer_shape_confidence(bars).where(is_hanging_man(bars), 0.0)


def shooting_star_confidence(bars: pd.DataFrame) -> pd.Series:
    return _inverted_hammer_shape_confidence(bars).where(is_shooting_star(bars), 0.0)


def inverted_hammer_confidence(bars: pd.DataFrame) -> pd.Series:
    return _inverted_hammer_shape_confidence(bars).where(is_inverted_hammer(bars), 0.0)


def _engulfing_degree(bars: pd.DataFrame) -> pd.Series:
    """How much the current candle's body exceeds the previous candle's --
    a bare/minimal engulf scores low, a decisive one scores high."""
    prev_open, prev_close = bars["open"].shift(1), bars["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    curr_body = (bars["close"] - bars["open"]).abs()
    size_ratio = curr_body / prev_body.replace(0, np.nan)
    return (_ramp_score(size_ratio, 1.0, 2.5) * 100).fillna(0)


def bullish_engulfing_confidence(bars: pd.DataFrame) -> pd.Series:
    return _engulfing_degree(bars).where(is_bullish_engulfing(bars), 0.0)


def bearish_engulfing_confidence(bars: pd.DataFrame) -> pd.Series:
    return _engulfing_degree(bars).where(is_bearish_engulfing(bars), 0.0)


def _star_confidence(bars: pd.DataFrame) -> pd.Series:
    """Longer first candle + smaller second candle + the third candle
    closing deeper into the first candle's body -> higher confidence."""
    c1_open, _, _, c1_close, _, c1_rng, _, _ = _components(bars, shift=2)
    c2_open, _, _, c2_close, c2_body, c2_rng, _, _ = _components(bars, shift=1)

    c1_rng_safe = c1_rng.replace(0, np.nan)
    c1_body = (c1_open - c1_close).abs()
    c1_score = _ramp_score(c1_body / c1_rng_safe, 0.5, 0.9)
    c2_score = _ramp_score(-(c2_body / c2_rng.replace(0, np.nan)), -0.3, 0.0)

    c1_midpoint = (c1_open + c1_close) / 2
    depth = (bars["close"] - c1_midpoint).abs() / c1_body.replace(0, np.nan)
    depth_score = _ramp_score(depth, 0.0, 1.0)

    return ((c1_score + c2_score + depth_score) / 3 * 100).fillna(0)


def morning_star_confidence(bars: pd.DataFrame) -> pd.Series:
    return _star_confidence(bars).where(is_morning_star(bars), 0.0)


def evening_star_confidence(bars: pd.DataFrame) -> pd.Series:
    return _star_confidence(bars).where(is_evening_star(bars), 0.0)


def _three_candle_confidence(bars: pd.DataFrame, bullish: bool) -> pd.Series:
    """Smaller wicks (stronger closes) + more consistent bar-over-bar
    progression -> higher confidence. Shared by soldiers/crows, mirrored
    by which side of the range the wick is measured from."""
    o0, h0, l0, c0, _, r0, _, _ = _components(bars, shift=2)
    o1, h1, l1, c1, _, r1, _, _ = _components(bars, shift=1)
    o2, h2, l2, c2, _, r2, _, _ = _components(bars, shift=0)

    if bullish:
        wick0, wick1, wick2 = (h0 - c0), (h1 - c1), (h2 - c2)
        progress = ((c1 - c0) + (c2 - c1)) / 2
    else:
        wick0, wick1, wick2 = (c0 - l0), (c1 - l1), (c2 - l2)
        progress = ((c0 - c1) + (c1 - c2)) / 2

    avg_wick_ratio = (
        wick0 / r0.replace(0, np.nan) + wick1 / r1.replace(0, np.nan) + wick2 / r2.replace(0, np.nan)
    ) / 3
    wick_score = _ramp_score(-avg_wick_ratio, -0.25, 0.0)
    progress_score = _ramp_score(progress / r1.replace(0, np.nan), 0.0, 0.5)

    return ((wick_score + progress_score) / 2 * 100).fillna(0)


def three_white_soldiers_confidence(bars: pd.DataFrame) -> pd.Series:
    return _three_candle_confidence(bars, bullish=True).where(is_three_white_soldiers(bars), 0.0)


def three_black_crows_confidence(bars: pd.DataFrame) -> pd.Series:
    return _three_candle_confidence(bars, bullish=False).where(is_three_black_crows(bars), 0.0)


def range_contraction(bars: pd.DataFrame, lookback: int = 20, quantile: float = 0.2) -> pd.Series:
    """True when the rolling `lookback`-bar high-low range is in the
    bottom `quantile` of its own trailing history -- a 'coiling' breakout
    setup. Symbol-relative (compares to this bars series' own trailing
    distribution), not a fixed absolute threshold. Uses an expanding
    (point-in-time) quantile rather than the full series' quantile so this
    is safe to use directly as a live trading signal -- no lookahead."""
    rng = bars["high"] - bars["low"]
    rolling_range = rng.rolling(lookback, min_periods=lookback).mean()
    threshold = rolling_range.expanding(min_periods=lookback).quantile(quantile)
    return (rolling_range <= threshold).fillna(False)
