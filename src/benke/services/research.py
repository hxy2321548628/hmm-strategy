"""Application service coordinating data, features, HMM, signals and evaluation."""

import logging
from dataclasses import dataclass

import pandas as pd

from benke.backtesting.metrics import performance
from benke.backtesting.timing import evaluate_signals
from benke.config.settings import AppConfig
from benke.data.repository import (
    FileMarketRepository,
    MarketRepository,
    align_market,
    validate_market,
)
from benke.exceptions import DataError, ModelError
from benke.features.builder import build_features
from benke.features.preprocessing import FeaturePipeline
from benke.models.hmm import filter_states, fit_hmm
from benke.strategies.signals import cash_states

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchResult:
    daily: pd.DataFrame
    windows: pd.DataFrame
    metrics: dict[str, float]


class ResearchService:
    def __init__(self, settings: AppConfig, repository: MarketRepository | None = None):
        self.settings = settings
        self.repository = repository or FileMarketRepository(settings.data)

    def run(self) -> ResearchResult:
        return self.run_frames(*self.repository.load_market())

    def run_frames(self, derivatives: pd.DataFrame, benchmark: pd.DataFrame) -> ResearchResult:
        config = self.settings
        derivatives, benchmark = align_market(
            validate_market(derivatives, ("close",)), validate_market(benchmark, ("close",))
        )
        features = build_features(derivatives, config.features)
        returns = benchmark.close.pct_change(fill_method=None)
        train_size, test_size = config.window.train_size, config.window.test_size
        if len(features) <= train_size:
            raise DataError("Not enough observations for training plus out-of-sample evaluation")
        records, diagnostics = [], []
        # Formulas are causal; computing them once avoids duplicated work and keeps warmup history.
        for start in range(train_size, len(features), test_size):
            stop = min(start + test_size, len(features))
            train = features.iloc[start - train_size : start].dropna()
            train = train.loc[returns.reindex(train.index).notna()]
            test = features.iloc[start:stop]
            try:
                pipeline = FeaturePipeline(config.features, config.preprocessing)
                train_values = pipeline.fit_transform(train)
                model = fit_hmm(train_values, config.hmm)
                train_states, posterior = filter_states(model, train_values)
                excluded = cash_states(
                    train_states, returns.loc[train.index], model.n_components, config.signal
                )
                test_states, _ = filter_states(model, pipeline.transform(test), posterior)
            except (ModelError, ValueError) as exc:
                raise ModelError(f"Window beginning {test.index[0].date()}: {exc}") from exc
            records.append(
                pd.DataFrame(
                    {
                        "state": test_states,
                        "signal": [int(state not in excluded) for state in test_states],
                    },
                    index=test.index,
                )
            )
            diagnostics.append(
                {
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "n_states": model.n_components,
                    "n_features": pipeline.n_components,
                    "cash_states": ",".join(map(str, sorted(excluded))),
                    "iterations": model.monitor_.iter,
                }
            )
            logger.info(
                "Window %s to %s: states=%s, features=%s",
                test.index[0].date(),
                test.index[-1].date(),
                model.n_components,
                pipeline.n_components,
            )
        predicted = pd.concat(records)
        expected = features.index[train_size:]
        if not predicted.index.equals(expected):
            raise ModelError("Rolling windows did not cover the out-of-sample calendar exactly")
        daily = evaluate_signals(predicted.signal, returns.loc[expected], config.backtest)
        daily["state"] = predicted.state
        metrics = performance(daily.strategy_return, config.backtest)
        metrics["rebalance_count"] = int((daily.turnover > 0).sum())
        metrics["benchmark_total_return"] = float(daily.benchmark_equity.iloc[-1] - 1)
        return ResearchResult(daily, pd.DataFrame(diagnostics), metrics)
