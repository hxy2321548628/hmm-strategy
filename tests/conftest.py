import numpy as np
import pandas as pd
import pytest

from benke.config.settings import AppConfig


@pytest.fixture
def market():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=125, name="date")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(dates))))
    derivatives = pd.DataFrame(
        {
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": 10000,
        },
        index=dates,
    )
    benchmark = pd.DataFrame({"close": prices}, index=dates)
    return derivatives, benchmark


@pytest.fixture
def config(tmp_path):
    return AppConfig.model_validate(
        {
            "data": {
                "derivatives": str(tmp_path / "futures.csv"),
                "benchmark": str(tmp_path / "index.csv"),
            },
            "features": [{"name": "return_3", "kind": "log_return", "window": 3}],
            "hmm": {"n_states": 2, "max_iterations": 30},
            "window": {"train_size": 60, "test_size": 20},
            "report": {"directory": str(tmp_path / "outputs")},
        }
    )
