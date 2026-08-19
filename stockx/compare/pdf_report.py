from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed to write a PDF

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from stockx.analysis.periods import find_best_worst_periods, format_period_bullets
from stockx.compare.orchestrator import ComparisonReport
from stockx.compare.report import to_dataframe
from stockx.config import REPORTS_DIR

pd.plotting.register_matplotlib_converters()

SUMMARY_COLUMNS = [
    "rank", "strategy", "sharpe_ratio", "sortino_ratio", "total_return",
    "annualized_return", "max_drawdown", "win_rate", "profit_factor", "num_trades",
]


def default_pdf_path(symbol: str, interval: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPORTS_DIR / f"{symbol}_{interval}_{timestamp}.pdf"


def generate_pdf_report(report: ComparisonReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        _add_summary_page(pdf, report)
        _add_price_equity_page(pdf, report)
        for result, metrics in report.results:
            best, worst = find_best_worst_periods(result, report.bars)
            _add_strategy_page(pdf, result, metrics, best, worst)


def _add_summary_page(pdf: PdfPages, report: ComparisonReport) -> None:
    df = to_dataframe(report)
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"StockX Strategy Comparison: {report.symbol} ({report.interval})", fontsize=16, y=0.97)
    fig.text(
        0.5, 0.93,
        f"Window: {report.start} -> {report.end}   |   Ranked by: {report.rank_metric}",
        ha="center", fontsize=10, color="dimgray",
    )

    if df.empty:
        fig.text(0.5, 0.5, "No strategies produced a result.", ha="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        return

    table_df = df[SUMMARY_COLUMNS].copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:,.4f}")

    ax_table = fig.add_axes([0.05, 0.55, 0.9, 0.3])
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    ax_bar = fig.add_axes([0.12, 0.08, 0.8, 0.38])
    plot_df = df.sort_values(report.rank_metric, ascending=True, na_position="first")
    values = plot_df[report.rank_metric].fillna(0)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    ax_bar.barh(plot_df["strategy"], values, color=colors)
    ax_bar.set_xlabel(report.rank_metric)
    ax_bar.set_title(f"{report.rank_metric} by strategy")
    ax_bar.axvline(0, color="black", linewidth=0.8)

    pdf.savefig(fig)
    plt.close(fig)


def _resample_daily(bars: pd.DataFrame) -> pd.DataFrame:
    daily = bars.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return daily.dropna()


def _plot_candlesticks(ax, daily_bars: pd.DataFrame, width_days: float = 0.6) -> None:
    # Muted grayscale, not red/green: the equity lines overlaid on top use a
    # tab10 palette that includes both a red and a green series
    # (rsi_reversion, vwap_reversion) -- colored candlesticks would visually
    # collide with those. Candles are background price context here, the
    # equity lines are the actual comparison being drawn.
    up_color, down_color = "#b5b5b5", "#6e6e6e"
    for ts, row in daily_bars.iterrows():
        x = mdates.date2num(ts)
        color = up_color if row["close"] >= row["open"] else down_color
        ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.6, zorder=1, alpha=0.8)
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"]) or (row["high"] * 0.0005)
        ax.add_patch(Rectangle(
            (x - width_days / 2, body_bottom), width_days, body_height,
            facecolor=color, edgecolor=color, zorder=1, alpha=0.8,
        ))

    x_values = mdates.date2num(daily_bars.index)
    x_pad = width_days * 2
    ax.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
    y_pad = (daily_bars["high"].max() - daily_bars["low"].min()) * 0.05
    ax.set_ylim(daily_bars["low"].min() - y_pad, daily_bars["high"].max() + y_pad)
    ax.xaxis_date()


def _add_price_equity_page(pdf: PdfPages, report: ComparisonReport) -> None:
    daily = _resample_daily(report.bars)
    if daily.empty:
        return

    fig, ax_price = plt.subplots(figsize=(11, 8.5))
    _plot_candlesticks(ax_price, daily)
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(
        f"{report.symbol}: Strategy Equity vs Buy & Hold (daily candles, interval={report.interval})"
    )

    ax_equity = ax_price.twinx()
    ax_equity.set_zorder(ax_price.get_zorder() + 1)
    ax_equity.patch.set_visible(False)  # keep candlesticks visible through the equity axes

    cmap = plt.get_cmap("tab10")
    for i, (result, _metrics) in enumerate(report.results):
        ax_equity.plot(
            result.equity_curve.index, result.equity_curve.values,
            color=cmap(i % 10), linewidth=1.3, label=result.strategy_name, zorder=3,
        )

    initial_capital = None
    if report.benchmark_result is not None:
        ax_equity.plot(
            report.benchmark_result.equity_curve.index, report.benchmark_result.equity_curve.values,
            color="black", linewidth=1.8, linestyle="--", label="buy_and_hold", zorder=4,
        )
        initial_capital = report.benchmark_result.initial_capital
    elif report.results:
        initial_capital = report.results[0][0].initial_capital

    if initial_capital is not None:
        ax_equity.axhline(initial_capital, color="gray", linewidth=0.7, linestyle=":", alpha=0.7, zorder=2)

    ax_equity.set_ylabel("Equity ($)")
    lines, labels = ax_equity.get_legend_handles_labels()
    if lines:
        ax_equity.legend(lines, labels, loc="upper left", fontsize=8)

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_price.tick_params(axis="x", rotation=30, labelsize=7)

    pdf.savefig(fig)
    plt.close(fig)


def _add_strategy_page(pdf: PdfPages, result, metrics, best, worst) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.08, 0.45, 0.87, 0.42])

    equity = result.equity_curve
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1)
    ax.set_title(f"{result.strategy_name} -- equity curve")
    ax.set_ylabel("Equity ($)")

    if best is not None:
        ax.axvspan(best.start, best.end, color="#2ca02c", alpha=0.2, label="Best period")
    if worst is not None:
        ax.axvspan(worst.start, worst.end, color="#d62728", alpha=0.2, label="Worst period")
    if best is not None or worst is not None:
        ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    header = (
        f"Sharpe {metrics.sharpe_ratio:.2f}  |  Sortino {metrics.sortino_ratio:.2f}  |  "
        f"Total return {metrics.total_return:+.2%}  |  Max drawdown {metrics.max_drawdown:.2%}  |  "
        f"Trades {metrics.num_trades}"
    )
    lines = [header, ""]
    lines += format_period_bullets(best, "Best")
    lines.append("")
    lines += format_period_bullets(worst, "Worst")

    fig.text(0.08, 0.38, "\n".join(lines), fontsize=8.5, family="monospace", va="top")

    pdf.savefig(fig)
    plt.close(fig)
