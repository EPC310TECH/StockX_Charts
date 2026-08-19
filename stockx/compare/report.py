from pathlib import Path

import pandas as pd

from stockx.compare.orchestrator import ComparisonReport


def to_dataframe(report: ComparisonReport) -> pd.DataFrame:
    rows = []
    best_value = None
    if report.results:
        best_value = getattr(report.results[0][1], report.rank_metric)

    for rank, (result, metrics) in enumerate(report.results, start=1):
        rank_value = getattr(metrics, report.rank_metric)
        delta = (
            rank_value - best_value
            if best_value is not None and not pd.isna(rank_value) and not pd.isna(best_value)
            else float("nan")
        )
        rows.append({
            "rank": rank,
            "strategy": result.strategy_name,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "num_trades": metrics.num_trades,
            f"{report.rank_metric}_delta_vs_best": delta,
        })

    if report.benchmark_metrics is not None:
        metrics = report.benchmark_metrics
        rank_value = getattr(metrics, report.rank_metric)
        delta = (
            rank_value - best_value
            if best_value is not None and not pd.isna(rank_value) and not pd.isna(best_value)
            else float("nan")
        )
        rows.append({
            "rank": "benchmark",
            "strategy": "buy_and_hold",
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "num_trades": metrics.num_trades,
            f"{report.rank_metric}_delta_vs_best": delta,
        })

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, report: ComparisonReport) -> None:
    print(f"\nStrategy comparison for {report.symbol} (interval: {report.interval})")
    print(f"  window: {report.start} -> {report.end}")
    print(f"  ranked by: {report.rank_metric}\n")

    if df.empty:
        print("  no strategies produced a result.")
    else:
        with pd.option_context("display.width", 120, "display.max_columns", None):
            print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        best = report.results[0]
        best_metrics = best[1]
        runner_up_gap = ""
        if len(report.results) > 1:
            gap = getattr(best_metrics, report.rank_metric) - getattr(report.results[1][1], report.rank_metric)
            runner_up_gap = f" (+{gap:.4f} {report.rank_metric} over runner-up)"
        print(f"\n  Best strategy: {report.best_strategy}{runner_up_gap}")

    if report.benchmark_metrics is not None:
        bm = report.benchmark_metrics
        print(
            f"\n  Buy & Hold: total return {bm.total_return:+.2%}, "
            f"annualized {bm.annualized_return:+.2%}, "
            f"{report.rank_metric} {getattr(bm, report.rank_metric):.4f}, "
            f"max drawdown {bm.max_drawdown:.2%}"
        )
        if report.results:
            best_value = getattr(report.results[0][1], report.rank_metric)
            bm_value = getattr(bm, report.rank_metric)
            if not pd.isna(best_value) and not pd.isna(bm_value):
                edge = best_value - bm_value
                verb = "beats" if edge >= 0 else "trails"
                print(f"  Best strategy ({report.best_strategy}) {verb} Buy & Hold by {abs(edge):.4f} {report.rank_metric}")

    if report.failed:
        print("\n  Strategies that failed to run:")
        for name, message in report.failed:
            print(f"    - {name}: {message}")


def save_report(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
