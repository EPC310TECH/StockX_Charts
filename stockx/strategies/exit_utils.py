import numpy as np
import pandas as pd

# Shared by every strategy's non-default exit_mode variants that need to
# "hold a position from its entry trigger until a later, different exit
# condition" -- e.g. RSI mean-reversion entering at RSI<30 but not exiting
# until RSI=50, rather than the instant RSI ticks back above 30. That's
# inherently stateful (today's position depends on what happened on prior
# bars, not just today's own indicator reading), unlike every strategy's
# *default* exit_mode, which stays a simple, stateless per-bar condition
# and is deliberately left as its own unchanged code path -- this helper
# is only reached by the new, opt-in variants.


def hold_until_exit(
    entry_long: pd.Series,
    entry_short: pd.Series,
    exit_long: pd.Series,
    exit_short: pd.Series,
) -> pd.Series:
    """State machine over four aligned boolean Series: from FLAT, an
    entry_long/entry_short bar opens a position; from LONG, an exit_long
    bar closes it back to FLAT (same for SHORT/exit_short). A bar that
    closes a position also re-checks for a fresh entry that same bar, so
    a direct LONG->SHORT flip in one bar is possible -- consistent with
    how every other strategy in this codebase (e.g. ma_crossover) already
    flips signals directly rather than forcing a FLAT bar in between.

    Returns a Signal-valued pd.Series aligned to the input index.
    """
    el = entry_long.to_numpy()
    es = entry_short.to_numpy()
    xl = exit_long.to_numpy()
    xs = exit_short.to_numpy()

    out = np.zeros(len(el), dtype=int)
    state = 0
    for i in range(len(el)):
        if state == 1 and xl[i]:
            state = 0
        elif state == -1 and xs[i]:
            state = 0
        if state == 0:
            if el[i]:
                state = 1
            elif es[i]:
                state = -1
        out[i] = state
    return pd.Series(out, index=entry_long.index, dtype=int)


def debounce_state(raw_state: pd.Series, confirm_bars: int) -> pd.Series:
    """Filters a raw +1/0/-1 state series (e.g. sign(fast_ema - slow_ema))
    so the *output* only moves to a new value once the raw input has held
    that value for `confirm_bars` consecutive bars -- a "wait for
    confirmation" exit variant for the crossover-style strategies, whose
    default exit is the instant the raw condition itself flips. With
    confirm_bars=1 this is identical to the raw input (no debounce), which
    is why it's safe to also use for the odd bar where a strategy wants
    "confirmed" behavior with a 1-bar confirmation window.
    """
    raw = raw_state.to_numpy()
    n = len(raw)
    out = np.zeros(n, dtype=int)
    current = 0
    run_value = raw[0] if n else 0
    run_length = 0
    for i in range(n):
        if raw[i] == run_value:
            run_length += 1
        else:
            run_value = raw[i]
            run_length = 1
        if run_length >= confirm_bars:
            current = run_value
        out[i] = current
    return pd.Series(out, index=raw_state.index, dtype=int)
