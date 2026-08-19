# StockX_Charts

A local tool for backtesting and comparing day-trading strategies on any ticker, with a candlestick/chart-pattern dashboard and PDF reporting. Personal-use project — data pulled via `yfinance`, cached locally, no server-side accounts or auth.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional -- only needed for tick/quote data (Alpaca/Databento)
```

## Usage

### Compare strategies on a symbol

```bash
python main.py AAPL --interval 1h
```

Runs every registered strategy against cached/fetched OHLCV bars, ranks them by Sharpe (or Sortino), prints a comparison table against buy & hold, and saves a PDF report to `reports/`.

Useful flags:

| Flag | Purpose |
|---|---|
| `--start` / `--end` | Date range (`YYYY-MM-DD`) |
| `--interval` | `1m`, `5m`, `15m`, `30m`, `1h`, `1d` (default `1m`) |
| `--strategies` | Comma-separated subset (default: all) |
| `--capital` | Initial capital (default `100000`) |
| `--commission` / `--slippage-bps` | Trade cost model |
| `--rank-metric` | `sharpe_ratio` or `sortino_ratio` |
| `--no-refresh` | Use only cached data, skip network fetch |
| `--output` | Save the comparison table as CSV |
| `--no-pdf` | Skip the PDF report |

### Walk-forward regime analysis

```bash
python main.py TSLA --interval 1h --walk-forward
```

Rolling out-of-sample folds, each tagged by the trend/volatility regime (`--regime-days` to classify, `--test-days` per test window) that preceded it — answers "does this strategy hold up out-of-sample, and in which regimes?" rather than just "did it win on this one window?" Needs enough history for multiple folds; `1h` (~2yr of data) is the recommended interval here, not `1m`.

### Candlestick/chart pattern scan

```bash
python main.py AAPL --patterns
```

Prints current pattern status for the symbol (confirmed candlestick patterns, forming multi-candle patterns, range contraction, recent chart-pattern breakouts) and regenerates the full pattern dashboard across the whole watchlist. **Not read-only** — this rewrites `reports/patterns_dashboard.html` and `reports/patterns_dashboard_data.json` every time it runs.

### Live dashboard

```bash
python main.py --serve --port 5000
```

Starts a local Flask dashboard (default port `5000`) with live symbol search, the pattern dashboard, saved watchlist, and chart layouts.

## Strategies

`orb`, `vwap_reversion`, `ma_crossover`, `rsi_reversion`, `volume_momentum`, `candlestick_reversal`, `range_breakout`, `chart_pattern_breakout`, `awesome_oscillator`, `macd_crossover`, `parabolic_sar`, `dual_thrust`, `heikin_ashi_trend`, `bollinger_reversion` — plus `buy_and_hold` as the fixed benchmark.

## Project layout

```
main.py                 CLI entrypoint
stockx/
  data/                 yfinance fetch + local parquet caching (data_cache/, tick_cache/)
  strategies/            strategy implementations, registered via @register
  backtest/engine.py     backtest engine
  metrics/                Sharpe/Sortino/drawdown/etc.
  analysis/               walk-forward, regime detection, pattern detection, scanner, forecasting
  compare/                report generation (CSV/PDF/HTML), watchlist, journal, Flask server
```

## Data sources

- **Bars (OHLCV):** `yfinance`, cached locally in `data_cache/` as parquet, respecting each interval's actual retention/request limits (see `stockx/config.py`).
- **Ticks/quotes (optional):** Alpaca (free tier, IEX feed) as the default; Databento as an opt-in premium source, only called when explicitly requested since it draws down a paid credit. Configure via `.env` (see `.env.example`).

## Notes

- `reports/`, `data_cache/`, `tick_cache/`, and `.env` are gitignored — all local/regenerable or secret, not tracked.
- Everything reads/writes locally; nothing is pushed to an external service beyond the market data fetch itself.
