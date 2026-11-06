"""Reproducible experiment grids. Failures are explicit, not zero-return successes."""

from concurrent.futures import ProcessPoolExecutor
from itertools import product

import pandas as pd

from benke.config.enums import ThresholdMode
from benke.config.settings import AppConfig
from benke.exceptions import BenkeError
from benke.services.research import ResearchService


def grid_settings(settings: AppConfig) -> list[dict]:
    grid = settings.search
    experiments = []
    for threshold, train, test, pca in product(
        grid.thresholds, grid.train_sizes, grid.test_sizes, grid.pca_variances
    ):
        payload = settings.model_dump(mode="json")
        payload["signal"].update(
            threshold_mode=ThresholdMode.FIXED.value, cumulative_threshold=threshold
        )
        payload["window"].update(train_size=train, test_size=test)
        payload["preprocessing"]["pca_variance"] = pca
        experiments.append(payload)
    return experiments


def _experiment(payload: dict) -> dict:
    row = {
        "threshold": payload["signal"]["cumulative_threshold"],
        "train_size": payload["window"]["train_size"],
        "test_size": payload["window"]["test_size"],
        "pca_variance": payload["preprocessing"]["pca_variance"],
    }
    try:
        # Pydantic validation is repeated after applying each experiment's overrides.
        settings = AppConfig.model_validate(payload)
        result = ResearchService(settings).run()
        return {**row, "status": "success", **result.metrics, "error": ""}
    except (BenkeError, ValueError, OSError) as exc:
        return {**row, "status": "failed", "error": str(exc)}


def run_search(settings: AppConfig) -> pd.DataFrame:
    experiments = grid_settings(settings)
    if settings.search.max_workers == 1:
        records = list(map(_experiment, experiments))
    else:
        with ProcessPoolExecutor(max_workers=settings.search.max_workers) as pool:
            records = list(pool.map(_experiment, experiments))
    return pd.DataFrame(records)
