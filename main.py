import argparse
import sys

from stockx.analysis.patterns import current_pattern_state, print_pattern_state
from stockx.analysis.walkforward import run_walk_forward
from stockx.backtest.engine import EngineConfig
from stockx.compare.orchestrator import run_comparison
from stockx.compare.patterns_html import update_patterns_dashboard
from stockx.compare.pdf_report import default_pdf_path, generate_pdf_report
from stockx.compare.report import print_report, save_report, to_dataframe
from stockx.compare.walkforward_report import (
    default_walkforward_pdf_path,
    generate_walkforward_pdf,
    print_walkforward_report,
)
from stockx.config import SUPPORTED_INTERVALS
from stockx.data.cache import get_bars
from stockx.exceptions import DataFetchError, InsufficientHistoryError
from stockx.strategies import get_registered_strategies, get_strategy_names


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest and compare day-trading strategies for a symbol."
    )
    parser.add_argument("symbol", nargs="?", default=None, help="Ticker symbol, e.g. AAPL (not required with --serve)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--interval",
        choices=SUPPORTED_INTERVALS,
        default="1m",
        help="Bar interval (default: 1m). Walk-forward mode needs enough "
        "history for multiple folds -- 1h (~2yr) is recommended there.",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help=f"Comma-separated subset of {{{', '.join(get_strategy_names())}}} (default: all)",
    )
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade leg ($)")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in basis points")
    parser.add_argument(
        "--rank-metric",
        choices=["sharpe_ratio", "sortino_ratio"],
        default="sharpe_ratio",
        help="Metric used to rank strategies (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip fetching fresh data from yfinance; use only what's already cached",
    )
    parser.add_argument("--output", default=None, help="Optional path to save the comparison report as CSV")
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip generating the PDF report (generated automatically by default)",
    )
    parser.add_argument(
        "--pdf-output", default=None, help="Optional path for the PDF report (default: reports/SYMBOL_INTERVAL_TIMESTAMP.pdf)"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward regime analysis (rolling out-of-sample folds tagged "
        "by trend/volatility regime) instead of a single-window comparison",
    )
    parser.add_argument(
        "--regime-days", type=int, default=20,
        help="Trading days used to classify regime before each test window (walk-forward mode)",
    )
    parser.add_argument(
        "--test-days", type=int, default=20,
        help="Trading days per out-of-sample test window (walk-forward mode)",
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Print current candlestick pattern status and save an interactive HTML "
        "pattern report, instead of running the full comparison",
    )
    parser.add_argument(
        "--pattern-forward-bars", type=int, default=10,
        help="Bars-ahead window used to compute each pattern's empirical win rate (--patterns mode)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the local pattern dashboard web server (enables live symbol lookup "
        "from the search box) instead of running any analysis",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for --serve (default: 5000)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.serve:
        from stockx import server
        print(f"Serving pattern dashboard at http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
        server.run(port=args.port)
        return 0

    if args.symbol is None:
        print("Error: symbol is required unless --serve is passed", file=sys.stderr)
        return 1

    if args.patterns:
        try:
            bars = get_bars(
                args.symbol, start=args.start, end=args.end,
                interval=args.interval, refresh=not args.no_refresh,
            )
        except (InsufficientHistoryError, DataFetchError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print_pattern_state(current_pattern_state(bars, args.symbol))

        html_path, symbol_count = update_patterns_dashboard(
            args.symbol, args.interval, bars,
            forward_bars=args.pattern_forward_bars,
        )
        print(f"\nSaved pattern dashboard to {html_path} ({symbol_count} symbols)")
        return 0

    strategies = None
    if args.strategies:
        requested = {name.strip() for name in args.strategies.split(",")}
        strategies = [s for s in get_registered_strategies() if s.name in requested]
        unknown = requested - {s.name for s in strategies}
        if unknown:
            print(f"Unknown strategy name(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(get_strategy_names())}", file=sys.stderr)
            return 1

    engine_config = EngineConfig(
        initial_capital=args.capital,
        commission_per_trade=args.commission,
        slippage_bps=args.slippage_bps,
    )

    if args.walk_forward:
        try:
            wf_report = run_walk_forward(
                args.symbol,
                interval=args.interval,
                regime_days=args.regime_days,
                test_days=args.test_days,
                strategies=strategies,
                engine_config=engine_config,
                rank_metric=args.rank_metric,
                refresh=not args.no_refresh,
            )
        except (InsufficientHistoryError, DataFetchError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print_walkforward_report(wf_report)

        if not args.no_pdf:
            pdf_path = args.pdf_output or default_walkforward_pdf_path(wf_report.symbol, wf_report.interval)
            generate_walkforward_pdf(wf_report, pdf_path)
            print(f"\nSaved PDF report to {pdf_path}")

        return 0

    try:
        report = run_comparison(
            args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            strategies=strategies,
            engine_config=engine_config,
            rank_metric=args.rank_metric,
            refresh=not args.no_refresh,
        )
    except (InsufficientHistoryError, DataFetchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    df = to_dataframe(report)
    print_report(df, report)

    if args.output:
        save_report(df, args.output)
        print(f"\nSaved report to {args.output}")

    if not args.no_pdf:
        pdf_path = args.pdf_output or default_pdf_path(report.symbol, report.interval)
        generate_pdf_report(report, pdf_path)
        print(f"\nSaved PDF report to {pdf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
