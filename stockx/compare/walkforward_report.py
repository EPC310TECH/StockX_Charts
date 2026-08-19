from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed to write a PDF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stockx.analysis.walkforward import WalkForwardReport, rank_bucket
from stockx.config import REPORTS_DIR


def default_walkforward_pdf_path(symbol: str, interval: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPORTS_DIR / f"{symbol}_{interval}_walkforward_{timestamp}.pdf"


def print_walkforward_report(report: WalkForwardReport) -> None:
    print(f"\nWalk-forward regime analysis for {report.symbol} (interval: {report.interval})")
    print(f"  window: {report.bars.index.min()} -> {report.bars.index.max()}")
    print(f"  folds: {len(report.folds)}   |   ranked by: {report.rank_metric}")

    trend, vol = report.current_regime
    print(f"\n  Current regime: {trend.upper()} / {vol.upper()}")

    if report.current_recommendation:
        bucket = report.regime_strategy_map.get(report.current_regime, {})
        stats = bucket.get(report.current_recommendation)
        if stats:
            avg_str = "N/A" if pd.isna(stats["avg_metric"]) else f"{stats['avg_metric']:.2f}"
            print(
                f"  Recommended strategy: {report.current_recommendation} "
                f"(won {stats['wins']}/{stats['folds']} folds in this regime, "
                f"avg {report.rank_metric} {avg_str})"
            )
        else:
            print(
                f"  Recommended strategy: {report.current_recommendation} "
                f"(no historical folds in this exact regime -- overall best shown)"
            )
    else:
        print("  Recommended strategy: none (not enough fold data)")

    if not report.folds:
        return

    print(f"\n  Regime bucket breakdown (wins/folds, avg {report.rank_metric}):")
    for key in sorted(report.regime_strategy_map.keys()):
        trend_r, vol_r = key
        bucket = report.regime_strategy_map[key]
        print(f"    {trend_r}/{vol_r}:")
        for name, stats in rank_bucket(bucket):
            avg_str = "N/A" if pd.isna(stats["avg_metric"]) else f"{stats['avg_metric']:.2f}"
            print(f"      {name:<18} {stats['wins']}/{stats['folds']} wins   avg {avg_str}")


def generate_walkforward_pdf(report: WalkForwardReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _add_summary_page(pdf, report)
        if report.folds:
            _add_timeline_page(pdf, report)


def _add_summary_page(pdf: PdfPages, report: WalkForwardReport) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"Walk-Forward Regime Analysis: {report.symbol} ({report.interval})", fontsize=16, y=0.97)
    window_str = f"{report.bars.index.min()} -> {report.bars.index.max()}"
    fig.text(
        0.5, 0.93,
        f"Window: {window_str}   |   Folds: {len(report.folds)}   |   Ranked by: {report.rank_metric}",
        ha="center", fontsize=10, color="dimgray",
    )

    trend, vol = report.current_regime
    rec = report.current_recommendation or "N/A"
    fig.text(
        0.5, 0.87,
        f"Current regime: {trend.upper()} / {vol.upper()}   ->   Recommended strategy: {rec}",
        ha="center", fontsize=12, weight="bold",
    )

    rows = []
    for key in sorted(report.regime_strategy_map.keys()):
        trend_r, vol_r = key
        bucket = report.regime_strategy_map[key]
        for name, stats in rank_bucket(bucket):
            avg_str = "N/A" if pd.isna(stats["avg_metric"]) else f"{stats['avg_metric']:.2f}"
            rows.append([f"{trend_r}/{vol_r}", name, f"{stats['wins']}/{stats['folds']}", avg_str])

    if not rows:
        fig.text(0.5, 0.5, "No folds produced results.", ha="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        return

    ax_table = fig.add_axes([0.08, 0.1, 0.84, 0.7])
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=rows,
        colLabels=["Regime", "Strategy", "Wins/Folds", f"Avg {report.rank_metric}"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    pdf.savefig(fig)
    plt.close(fig)


def _shade_regions(ax, index, mask, color, alpha) -> None:
    in_region = False
    start = None
    for ts, val in zip(index, mask):
        if val and not in_region:
            in_region, start = True, ts
        elif not val and in_region:
            in_region = False
            ax.axvspan(start, ts, color=color, alpha=alpha)
    if in_region:
        ax.axvspan(start, index[-1], color=color, alpha=alpha)


def _add_timeline_page(pdf: PdfPages, report: WalkForwardReport) -> None:
    bars = report.bars
    regime_series = report.regime_series

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[3, 1])

    ax1.plot(regime_series.index, regime_series["trend_strength"], color="#1f77b4", linewidth=0.8, label="ADX")
    ax1b = ax1.twinx()
    ax1b.plot(
        regime_series.index, regime_series["volatility_pct"] * 100,
        color="#ff7f0e", linewidth=0.8, alpha=0.7, label="ATR %",
    )
    ax1.set_ylabel("ADX")
    ax1b.set_ylabel("ATR %")
    ax1.set_title(f"{report.symbol} regime timeline (green shading = trending)")

    trending_mask = regime_series["trend_regime"] == "trending"
    _shade_regions(ax1, regime_series.index, trending_mask, color="#2ca02c", alpha=0.08)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    strategy_names = sorted({fold.winner for fold in report.folds if fold.winner})
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(strategy_names)}
    tz = bars.index.tz

    for fold in report.folds:
        if fold.winner is None:
            continue
        start = pd.Timestamp(fold.test_start).tz_localize(tz)
        end = pd.Timestamp(fold.test_end).tz_localize(tz) + pd.Timedelta(days=1)
        ax2.axvspan(start, end, color=color_map[fold.winner], alpha=0.7)

    ax2.set_yticks([])
    ax2.set_xlabel("Time")
    ax2.set_title("Winning strategy per fold")
    if strategy_names:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[name]) for name in strategy_names]
        ax2.legend(handles, strategy_names, loc="upper left", fontsize=7, ncol=min(len(strategy_names), 3))

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    pdf.savefig(fig)
    plt.close(fig)
