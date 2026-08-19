import pandas as pd

from stockx.strategies.base import Signal, Strategy

# Deliberately NOT @register-decorated: this isn't a candidate strategy to
# rank against the others, it's a fixed benchmark run separately (with
# flatten_eod=False, since it must hold positions overnight/across days
# unlike every registered day-trading strategy). Registering it would let
# it be silently mis-run through the normal --strategies flow with
# flatten_eod=True, breaking its entire purpose.


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    display_name = "Buy & Hold"

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(Signal.LONG, index=bars.index, dtype=int)
