import math
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from stockx.strategies.base import Signal, Strategy
from stockx.strategies.indicators import infer_bar_minutes, session_groups

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    side: int  # 1 long, -1 short
    qty: float
    pnl: float
    return_pct: float
    commission: float
    slippage: float


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    params: dict
    equity_curve: pd.Series
    positions: pd.Series
    trades: List[Trade]
    initial_capital: float
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class EngineConfig:
    initial_capital: float = 100_000.0
    position_size_pct: float = 1.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 1.0
    flatten_eod: bool = True
    eod_buffer_minutes: int = 1
    # "next_open": a signal decided from bar N's close fills at bar N+1's
    #   open (default -- no lookahead, matches a real order queued after the
    #   bar that triggered it).
    # "same_close": fills at bar N's own close, the instant the signal was
    #   decided -- optimistic (assumes you can trade at the exact price your
    #   decision was based on) but useful for fast strategy comparison.
    execution_timing: str = "next_open"
    # Resolves which of a stop-loss/take-profit level fires first when both
    # fall inside one bar's high-low range -- OHLC bars alone don't record
    # the true tick order. Only matters for strategies whose
    # generate_stop_target returns levels; a no-op otherwise.
    # "ohlc": assume the bar's path was open -> high -> low -> close.
    # "olhc": assume open -> low -> high -> close.
    intrabar_path: str = "ohlc"


class BacktestEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def run(self, strategy: Strategy, bars: pd.DataFrame, symbol: str) -> BacktestResult:
        self._validate(bars)
        cfg = self.config
        if cfg.execution_timing not in ("next_open", "same_close"):
            raise ValueError(f"unknown execution_timing {cfg.execution_timing!r}")
        if cfg.intrabar_path not in ("ohlc", "olhc"):
            raise ValueError(f"unknown intrabar_path {cfg.intrabar_path!r}")

        raw_signals = strategy.generate_signals(bars)
        if cfg.flatten_eod:
            raw_signals = self._apply_eod_flatten(raw_signals, bars)

        if cfg.execution_timing == "same_close":
            # The signal decided from bar N's own close fills at that same
            # bar's close -- no shift, no wait for the next bar.
            target_position = raw_signals.fillna(0).astype(int)
            fill_prices = bars["close"]
        else:
            # Fills happen at the next bar's open -> shift signal by one bar
            # so today's decision is never used to fill today's bar (no
            # lookahead).
            target_position = raw_signals.shift(1).fillna(0).astype(int)
            fill_prices = bars["open"]

        stop_target = strategy.generate_stop_target(bars)

        trades, equity_curve = self._simulate(
            bars, target_position, symbol, strategy.name, fill_prices, stop_target
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            params=dict(strategy.params),
            equity_curve=equity_curve,
            positions=target_position,
            trades=trades,
            initial_capital=cfg.initial_capital,
            start=bars.index[0],
            end=bars.index[-1],
        )

    def _validate(self, bars: pd.DataFrame) -> None:
        if bars.empty:
            raise ValueError("bars is empty")
        missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"bars missing required columns: {missing}")
        if not bars.index.is_monotonic_increasing:
            raise ValueError("bars index must be sorted ascending")
        if bars.index.tz is None:
            raise ValueError("bars index must be tz-aware")

    def _apply_eod_flatten(self, signals: pd.Series, bars: pd.DataFrame) -> pd.Series:
        signals = signals.copy()
        buffer_bars = max(1, round(self.config.eod_buffer_minutes / infer_bar_minutes(bars)))
        for _, group in session_groups(bars):
            # A one-bar "session" (daily-or-coarser interval, where each bar
            # already spans a full trading day) has no intraday close to
            # flatten ahead of -- treating it as its own last-N-bars would
            # flatten literally every bar, silently zeroing every strategy's
            # signal regardless of what generate_signals produced.
            if len(group.index) <= 1:
                continue
            n = min(buffer_bars, len(group.index))
            signals.loc[group.index[-n:]] = Signal.FLAT
        return signals

    def _simulate(self, bars, target_position, symbol, strategy_name, fill_prices, stop_target=None):
        cfg = self.config
        highs = bars["high"]
        lows = bars["low"]
        closes = bars["close"]
        idx = bars.index
        n = len(idx)
        slip = cfg.slippage_bps / 10_000.0

        run_id = (target_position != target_position.shift(1)).cumsum()
        run_groups = target_position.groupby(run_id)
        run_positions = run_groups.first()
        run_starts = run_groups.apply(lambda g: g.index[0])
        run_ends = run_groups.apply(lambda g: g.index[-1])

        pos_loc = {ts: i for i, ts in enumerate(idx)}

        trades: List[Trade] = []
        equity = pd.Series(cfg.initial_capital, index=idx, dtype=float)
        realized_pnl = 0.0

        for rid in run_positions.index:
            side = int(run_positions.loc[rid])
            start_i = pos_loc[run_starts.loc[rid]]
            end_i = pos_loc[run_ends.loc[rid]]

            # Size against *current* capital (starting capital plus realized
            # P&L so far), not the fixed starting amount -- otherwise every
            # trade is re-sized as if the account had never lost (or gained)
            # a cent, letting losses compound without ever shrinking
            # position size. Once current capital is exhausted, there's
            # nothing left to trade with: freeze equity there rather than
            # let it spiral past zero, which also keeps pct_change()-based
            # metrics (Sharpe/Sortino) well-defined -- dividing by a
            # negative prior-equity value flips the sign of every
            # subsequent "return", making them meaningless.
            current_capital = cfg.initial_capital + realized_pnl

            if side == 0 or current_capital <= 0:
                equity.iloc[start_i:end_i + 1] = current_capital
                continue

            entry_price = float(fill_prices.iloc[start_i]) * (1 + slip * side)
            qty = math.floor((current_capital * cfg.position_size_pct) / entry_price)
            if qty <= 0:
                equity.iloc[start_i:end_i + 1] = current_capital
                continue
            entry_commission = cfg.commission_per_trade
            base_equity = current_capital - entry_commission

            # A bracket stop-loss/take-profit, if this strategy defines one,
            # is read once at entry and held fixed for the run's duration --
            # not recomputed bar-to-bar. Walk the run's bars checking each
            # one's high/low for a touch; the first bar (if any) where the
            # position would be stopped or targeted out overrides the
            # run's own end_i for exit purposes.
            stop_price = target_price = None
            if stop_target is not None:
                row = stop_target.iloc[start_i]
                sp, tp = row.get("stop_price"), row.get("target_price")
                stop_price = float(sp) if pd.notna(sp) else None
                target_price = float(tp) if pd.notna(tp) else None

            bracket_exit_i = None
            bracket_exit_price = None
            if stop_price is not None or target_price is not None:
                for i in range(start_i, end_i + 1):
                    bar_high, bar_low = float(highs.iloc[i]), float(lows.iloc[i])
                    hit_stop = stop_price is not None and (
                        bar_low <= stop_price if side == 1 else bar_high >= stop_price
                    )
                    hit_target = target_price is not None and (
                        bar_high >= target_price if side == 1 else bar_low <= target_price
                    )
                    if hit_stop and hit_target:
                        # Both levels sit inside this one bar's high-low
                        # range -- OHLC data alone doesn't record which tick
                        # came first, so resolve it via the configured
                        # intrabar path instead of guessing. A long's stop
                        # sits on the low side and target on the high side
                        # (and vice-versa for a short); whichever level is on
                        # the side the path assumption reaches first wins.
                        high_reached_first = cfg.intrabar_path == "ohlc"
                        stop_is_upper_side = side == -1
                        if stop_is_upper_side == high_reached_first:
                            bracket_exit_i, bracket_exit_price = i, stop_price
                        else:
                            bracket_exit_i, bracket_exit_price = i, target_price
                    elif hit_stop:
                        bracket_exit_i, bracket_exit_price = i, stop_price
                    elif hit_target:
                        bracket_exit_i, bracket_exit_price = i, target_price
                    if bracket_exit_i is not None:
                        break

            fill_end_i = bracket_exit_i if bracket_exit_i is not None else end_i
            for i in range(start_i, fill_end_i + 1):
                unrealized = qty * side * (float(closes.iloc[i]) - entry_price)
                # A single position -- especially a short, whose risk is
                # unbounded -- can still lose more than the capital it was
                # sized against within one holding period even though it
                # was sized correctly at entry. A real account gets force-
                # liquidated at zero, not carried net-negative; floor the
                # displayed equity path the same way.
                equity.iloc[i] = max(0.0, base_equity + unrealized)

            if bracket_exit_i is not None:
                exit_price = bracket_exit_price * (1 - slip * side)
                exit_time = idx[bracket_exit_i]
            elif end_i + 1 < n:
                exit_price = float(fill_prices.iloc[end_i + 1]) * (1 - slip * side)
                exit_time = idx[end_i + 1]
            else:
                exit_price = float(closes.iloc[end_i])
                exit_time = idx[end_i]

            exit_commission = cfg.commission_per_trade
            # The Trade record keeps the true, uncapped pnl/return_pct --
            # an honest account of what that trade actually did -- but the
            # capital carried forward into position sizing for the *next*
            # trade can't drop below total wipeout (-initial_capital).
            pnl = qty * side * (exit_price - entry_price) - entry_commission - exit_commission
            realized_pnl = max(realized_pnl + pnl, -cfg.initial_capital)
            # Position-relative, not account-capital-relative: this trade's
            # own return on the notional it deployed (qty * entry_price),
            # matching the usual "trade return %" convention in backtest
            # reports. Only differs from a capital-relative figure when
            # position_size_pct < 1.0 -- not currently reachable from the
            # web UI (always 1.0 there), so the two coincide in practice
            # today, but this stays position-relative if that changes.
            return_pct = (pnl / (qty * entry_price)) if qty > 0 else 0.0

            if bracket_exit_i is not None and bracket_exit_i < end_i:
                # Stopped/targeted out before the strategy's own signal
                # changed -- stay flat for the rest of this run rather than
                # silently re-entering, since nothing in the signal stream
                # told the engine to.
                equity.iloc[bracket_exit_i + 1:end_i + 1] = cfg.initial_capital + realized_pnl

            trades.append(Trade(
                symbol=symbol,
                strategy=strategy_name,
                entry_time=run_starts.loc[rid],
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                side=side,
                qty=qty,
                pnl=pnl,
                return_pct=return_pct,
                commission=entry_commission + exit_commission,
                slippage=slip,
            ))

        return trades, equity
