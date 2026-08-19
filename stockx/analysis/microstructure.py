from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from stockx.data import ticks

# Microstructure-derived forecasting signals from real Alpaca tick data --
# the "genuinely different, less publicly arbitraged" signal class flagged
# during the earlier research brainstorm. Every classic-TA signal tested
# so far (candlestick patterns, RSI/MACD/Bollinger/Stochastic/AO/SAR,
# volume/money-flow indicators) is built from daily/hourly OHLC bars that
# any retail charting platform already shows -- and showed no real
# out-of-sample edge across single-symbol, cross-sectional, and
# multi-horizon tests. Order-flow and order-book imbalance are different
# in kind: they're documented in the market-microstructure literature
# (e.g. Cont, Kukanov & Stoikov 2014) to carry genuine short-horizon
# predictive power for price direction -- but typically at horizons of
# seconds to a few minutes, which is why this operates on tick-bucketed
# intraday windows, not daily bars, and needs real trade/quote data (not
# OHLCV) to compute at all.
#
# No walk-forward train/test split here, unlike forecast.py -- there's no
# fitted weight to leak information into. Each feature is a raw, directly
# observable quantity (not a trained combination), and both the feature
# and the forward return it's checked against are computed causally
# (feature uses data up to a window's close; forward return looks from
# that close to a later window's close), so a plain in-sample IC is a
# legitimate, standard way to evaluate it -- exactly how the
# microstructure literature itself evaluates these quantities.


def classify_trade_side(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.Series:
    """Lee-Ready style trade classification: a trade above the prevailing
    quote midpoint is buyer-initiated (+1), below is seller-initiated
    (-1). A trade exactly at the midpoint falls back to the tick rule
    (compare to the previous trade's own price). Each trade is matched to
    the most recent quote at or before its own timestamp (merge_asof,
    direction="backward") -- never a quote that arrives after it, so
    there's no lookahead in the classification itself.
    """
    quote_mid = pd.DataFrame({"mid": (quotes["bid_price"] + quotes["ask_price"]) / 2})
    merged = pd.merge_asof(
        trades[["price"]].sort_index(), quote_mid.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    side = np.sign(merged["price"].to_numpy() - merged["mid"].to_numpy())
    tick_side = np.sign(merged["price"].diff().fillna(0.0).to_numpy())
    side = np.where(side == 0, tick_side, side)
    return pd.Series(side, index=trades.index, name="side")


def compute_microstructure_features(trades: pd.DataFrame, quotes: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Bucket raw ticks into `freq` windows and derive, per window:
    - order_flow_imbalance: (buy volume - sell volume) / total volume from
      classified trades, in [-1, 1]
    - quote_imbalance: average (bid_size - ask_size)/(bid_size+ask_size)
      across quote updates in the window ("book pressure"), in [-1, 1]
    - spread_pct: average relative bid-ask spread, a liquidity/uncertainty gauge
    - close: the window's last trade price (forward returns are computed from this)
    - trade_count / quote_count: activity gauges, used to drop too-thin
      windows before trusting a feature computed from them
    """
    side = classify_trade_side(trades, quotes)
    signed_volume = side * trades["size"]

    trade_bucket = trades.index.floor(freq)
    total_volume = trades["size"].groupby(trade_bucket).sum()
    ofi = signed_volume.groupby(trade_bucket).sum() / total_volume.replace(0, np.nan)
    trade_count = trades["size"].groupby(trade_bucket).count()
    close = trades["price"].groupby(trade_bucket).last()

    quote_bucket = quotes.index.floor(freq)
    book_imbalance = (quotes["bid_size"] - quotes["ask_size"]) / (quotes["bid_size"] + quotes["ask_size"]).replace(0, np.nan)
    quote_imbalance = book_imbalance.groupby(quote_bucket).mean()
    mid = (quotes["bid_price"] + quotes["ask_price"]) / 2
    spread_pct = ((quotes["ask_price"] - quotes["bid_price"]) / mid).groupby(quote_bucket).mean()
    quote_count = quotes["bid_price"].groupby(quote_bucket).count()

    trade_features = pd.DataFrame({"order_flow_imbalance": ofi, "trade_count": trade_count, "close": close})
    quote_features = pd.DataFrame({"quote_imbalance": quote_imbalance, "spread_pct": spread_pct, "quote_count": quote_count})
    return trade_features.join(quote_features, how="outer").sort_index()


@dataclass
class MicrostructureSignalReport:
    symbol: str
    freq: str
    forward_windows: int
    n_windows: int
    ic_by_feature: Dict[str, float]


def validate_microstructure_signals(
    symbol: str,
    start: datetime,
    end: datetime,
    freq: str = "15min",
    forward_windows: int = 1,
    min_trades_per_window: int = 5,
    min_quotes_per_window: int = 20,
) -> MicrostructureSignalReport:
    trades = ticks.fetch_alpaca_trades(symbol, start, end)
    quotes = ticks.fetch_alpaca_quotes(symbol, start, end)
    features = compute_microstructure_features(trades, quotes, freq=freq)

    thin = (features["trade_count"].fillna(0) < min_trades_per_window) | (
        features["quote_count"].fillna(0) < min_quotes_per_window
    )
    features = features[~thin]

    forward_return = features["close"].shift(-forward_windows) / features["close"] - 1

    ic_by_feature: Dict[str, float] = {}
    for feature_name in ("order_flow_imbalance", "quote_imbalance", "spread_pct"):
        valid = features[feature_name].notna() & forward_return.notna()
        if valid.sum() < 10:
            ic_by_feature[feature_name] = float("nan")
            continue
        ic_by_feature[feature_name] = float(
            features[feature_name][valid].rank().corr(forward_return[valid].rank())
        )

    return MicrostructureSignalReport(
        symbol=symbol.upper(),
        freq=freq,
        forward_windows=forward_windows,
        n_windows=int((features["close"].notna() & forward_return.notna()).sum()),
        ic_by_feature=ic_by_feature,
    )
