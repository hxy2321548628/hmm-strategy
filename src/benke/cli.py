"""CLI is the composition root; business modules never parse command-line arguments."""

import argparse
import json
import logging
import sys
from pathlib import Path

from benke.config.settings import Settings, load_settings
from benke.exceptions import BenkeError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HMM market timing research")
    parser.add_argument("command", choices=["validate", "run", "portfolio", "search", "schema"])
    parser.add_argument("--config", type=Path, help="TOML config (required except for schema)")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Settings.model_json_schema(), ensure_ascii=False, indent=2))
        return 0
    if args.config is None:
        parser.error("--config is required")
    try:
        settings = load_settings(args.config)
        logging.basicConfig(
            level=settings.log_level.value, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        from benke.data.repository import FileMarketRepository
        from benke.reporting.export import create_run_directory, save_research, write_metrics
        from benke.services.research import ResearchService

        if args.command == "validate":
            derivatives, benchmark = FileMarketRepository(settings.data).load_market()
            from benke.features.builder import build_features
            from benke.features.preprocessing import FeaturePipeline

            features = build_features(derivatives, settings.features)
            if len(features) <= settings.window.train_size:
                raise ValueError("Not enough data for an out-of-sample window")
            pipeline = FeaturePipeline(settings.features, settings.preprocessing)
            pipeline.fit_transform(features.iloc[: settings.window.train_size].dropna())
            print(f"Configuration and market data valid: {len(benchmark)} aligned sessions")
            return 0
        directory = create_run_directory(settings, args.command)
        if args.command == "search":
            from benke.services.search import run_search

            result = run_search(settings)
            result.to_csv(directory / "search.csv", index=False)
            print(directory)
            return 1 if result.status.eq("failed").any() else 0
        research = ResearchService(settings).run()
        save_research(research, settings, directory)
        if args.command == "portfolio":
            from benke.backtesting.portfolio import run_portfolio
            from benke.config.enums import WeightingMode

            repository = FileMarketRepository(settings.data)
            portfolio = run_portfolio(
                research.daily.signal,
                repository.load_stocks(),
                repository.load_weights(equal=settings.backtest.weighting == WeightingMode.EQUAL),
                settings.backtest,
            )
            portfolio.daily.to_csv(directory / "portfolio_daily.csv", index_label="date")
            portfolio.orders.to_csv(directory / "orders.csv", index=False)
            write_metrics(portfolio.metrics, directory / "portfolio_metrics.json")
        print(directory)
        return 0
    except (BenkeError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
