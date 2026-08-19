import numpy as np
import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def _true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    tr = _true_range(bars)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_pct(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of price -- a scale-free volatility measure,
    comparable across symbols with very different price levels."""
    return average_true_range(bars, period) / bars["close"]


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index -- trend-strength indicator,
    independent of trend direction (a strong downtrend scores as high as a
    strong uptrend)."""
    up_move = bars["high"].diff()
    down_move = -bars["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=bars.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=bars.index
    )

    atr = average_true_range(bars, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0)


def money_flow_index(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Volume-weighted RSI: same overbought/oversold reading as RSI, but
    built from typical-price * volume rather than price alone, so it
    reflects conviction (volume) behind a move, not just price direction."""
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    raw_flow = typical_price * bars["volume"]
    rising = typical_price.diff() > 0

    positive_flow = raw_flow.where(rising, 0.0).rolling(period, min_periods=period).sum()
    negative_flow = raw_flow.where(~rising, 0.0).rolling(period, min_periods=period).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    result = 100 - (100 / (1 + money_ratio))
    return result.fillna(50)


def on_balance_volume(bars: pd.DataFrame) -> pd.Series:
    """Running total of volume, added on up closes and subtracted on down
    closes -- a cumulative read on whether volume is backing the
    accumulation or distribution side of price action."""
    direction = np.sign(bars["close"].diff()).fillna(0.0)
    return (direction * bars["volume"]).cumsum()


def chaikin_money_flow(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume-weighted read on buying vs. selling pressure: where in each
    bar's own high-low range the close sits (near the high = buying
    pressure, near the low = selling pressure), weighted by that bar's
    volume and summed over a rolling window. Positive = net buying
    pressure, negative = net selling."""
    rng = (bars["high"] - bars["low"]).replace(0, np.nan)
    money_flow_multiplier = ((bars["close"] - bars["low"]) - (bars["high"] - bars["close"])) / rng
    money_flow_volume = money_flow_multiplier.fillna(0.0) * bars["volume"]
    return (
        money_flow_volume.rolling(period, min_periods=period).sum()
        / bars["volume"].rolling(period, min_periods=period).sum().replace(0, np.nan)
    ).fillna(0.0)


def volume_surge(bars: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.Series:
    """True when a bar's volume is well above its own recent average --
    relative to the symbol's own trailing volume, not a fixed share count,
    so it means the same thing across symbols of very different liquidity."""
    avg_volume = bars["volume"].rolling(period, min_periods=period).mean()
    return bars["volume"] > (multiplier * avg_volume)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(bars: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """(%K, %D)."""
    low_min = bars["low"].rolling(k_period, min_periods=k_period).min()
    high_max = bars["high"].rolling(k_period, min_periods=k_period).max()
    percent_k = 100 * (bars["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    percent_d = percent_k.rolling(d_period, min_periods=d_period).mean()
    return percent_k, percent_d


def awesome_oscillator(bars: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Bill Williams' Awesome Oscillator: SMA(fast) - SMA(slow) of the
    bar's median price (high+low)/2 -- momentum relative to its own recent
    average, not to price itself."""
    median_price = (bars["high"] + bars["low"]) / 2
    return median_price.rolling(fast).mean() - median_price.rolling(slow).mean()


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """(upper, mid, lower)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def parabolic_sar(bars: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Wilder's Parabolic SAR (stop-and-reverse). Recursive by definition
    (each bar's SAR depends on the prior bar's SAR/extreme-point/
    acceleration-factor state), so unlike every other indicator in this
    module it can't be expressed as a vectorized rolling/ewm operation --
    computed with an explicit bar-by-bar loop instead, same as the
    backtest engine's own per-run simulation loop.

    Returns a pd.Series aligned to bars.index: the SAR price for each bar,
    trailing below price in an uptrend and above price in a downtrend.
    """
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    n = len(bars)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=bars.index)

    uptrend = highs[1] >= highs[0]
    sar[0] = lows[0] if uptrend else highs[0]
    ep = highs[0] if uptrend else lows[0]
    af = af_step

    for i in range(1, n):
        prev_sar = sar[i - 1]
        candidate = prev_sar + af * (ep - prev_sar)

        if uptrend:
            # SAR can never sit inside the prior two bars' range.
            candidate = min(candidate, lows[i - 1], lows[i - 2] if i >= 2 else lows[i - 1])
            if lows[i] < candidate:
                uptrend = False
                sar[i] = ep
                ep = lows[i]
                af = af_step
            else:
                sar[i] = candidate
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            candidate = max(candidate, highs[i - 1], highs[i - 2] if i >= 2 else highs[i - 1])
            if highs[i] > candidate:
                uptrend = True
                sar[i] = ep
                ep = highs[i]
                af = af_step
            else:
                sar[i] = candidate
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=bars.index)


def heikin_ashi(bars: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi transform: smoothed synthetic candles that filter out
    noise for trend-following. HA close is the bar's own average; HA open
    is recursive (average of the *previous* HA candle's open and close),
    which is what gives Heikin-Ashi its characteristic smoothing --
    another recursive definition that can't be vectorized, computed with
    an explicit loop like parabolic_sar above.
    """
    ha_close = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4
    opens = bars["open"].to_numpy()
    closes = ha_close.to_numpy()
    n = len(bars)
    ha_open = np.empty(n)
    ha_open[0] = opens[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + closes[i - 1]) / 2
    ha_open = pd.Series(ha_open, index=bars.index)

    ha_high = pd.concat([bars["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([bars["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})


def infer_bar_minutes(bars: pd.DataFrame) -> float:
    """Modal spacing between consecutive bars, in minutes. Mode (not
    mean/median) is used so it isn't skewed by the large overnight gap
    between the last bar of one session and the first bar of the next."""
    deltas = bars.index.to_series().diff().dt.total_seconds() / 60
    deltas = deltas[deltas > 0]
    return float(deltas.mode().iloc[0]) if not deltas.empty else 1.0


def session_groups(bars: pd.DataFrame):
    """Group bars by trading day."""
    return bars.groupby(bars.index.date, sort=False)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """VWAP that resets at the start of each trading day."""
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical_price * bars["volume"]

    day = bars.index.date
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = bars["volume"].groupby(day).cumsum()
    return (cum_pv / cum_vol.replace(0, np.nan)).ffill()


def opening_range(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Per-day opening-range high/low, computed from the first `minutes`
    minutes of each trading day (converted to a bar count using the bars'
    own spacing, so this means the same thing at 1m or 15m resolution) and
    broadcast across that whole day's rows."""
    day = bars.index.date
    n_bars = max(1, round(minutes / infer_bar_minutes(bars)))

    def _per_day(group: pd.DataFrame) -> pd.DataFrame:
        window = group.iloc[:n_bars]
        or_high = window["high"].max()
        or_low = window["low"].min()
        return pd.DataFrame(
            {"or_high": or_high, "or_low": or_low}, index=group.index
        )

    return bars.groupby(day, sort=False, group_keys=False).apply(_per_day)
