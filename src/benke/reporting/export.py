"""Persist auditable results in a unique run directory, never overwrite a prior run."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from benke.config.settings import AppConfig
from benke.services.research import ResearchResult


def create_run_directory(settings: AppConfig, kind: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = settings.report.directory / f"{kind}-{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return directory


def write_metrics(metrics: dict, path: Path) -> None:
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def save_research(result: ResearchResult, settings: AppConfig, directory: Path) -> None:
    result.daily.to_csv(directory / "daily.csv", index_label="date")
    result.windows.to_csv(directory / "windows.csv", index=False)
    write_metrics(result.metrics, directory / "metrics.json")
    if settings.report.plot:
        plot_equity(result.daily, settings, directory / "equity.png")


def plot_equity(daily, settings: AppConfig, path: Path) -> None:
    # Local figure creation avoids changing global pyplot styles during package import.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=settings.report.figure_size)
    FigureCanvasAgg(figure)
    equity_axis, drawdown_axis = figure.subplots(2, 1, sharex=True)
    for column in ("strategy_equity", "benchmark_equity"):
        if column in daily:
            equity_axis.plot(daily.index, daily[column], label=column)
    equity_axis.legend()
    equity_axis.set_ylabel("Equity")
    equity = daily.strategy_equity
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    drawdown_axis.fill_between(daily.index, drawdown, alpha=0.3)
    drawdown_axis.set_ylabel("Drawdown")
    figure.tight_layout()
    figure.savefig(path, dpi=settings.report.dpi)
