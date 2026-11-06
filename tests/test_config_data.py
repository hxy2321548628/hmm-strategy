from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from benke.config.settings import AppConfig, load_settings
from benke.data.repository import align_market, read_table, validate_market
from benke.exceptions import DataError


def test_environment_paths_and_precedence(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text("""[data]
derivatives = "futures.csv"
benchmark = "index.csv"
[[features]]
name = "ret"
kind = "log_return"
[window]
train_size = 520
test_size = 15
""")
    monkeypatch.setenv("BENKE_WINDOW__TRAIN_SIZE", "100")
    monkeypatch.setenv("BENKE_REPORT__DIRECTORY", "results")
    monkeypatch.chdir("/")
    config = load_settings(path)
    assert config.window.train_size == 100
    assert config.window.test_size == 15
    assert config.report.directory == tmp_path / "results"
    payload = config.model_dump()
    payload["window"]["train_size"] = 120
    assert AppConfig.model_validate(payload).window.train_size == 120


@pytest.mark.parametrize("key,value", [("test_size", 0), ("train_size", -1), ("typo", 10)])
def test_invalid_window(config, key, value):
    payload = config.model_dump()
    payload["window"][key] = value
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_invalid_feature_and_duplicate(config):
    payload = config.model_dump()
    payload["features"] = [payload["features"][0]] * 2
    with pytest.raises(ValidationError, match="unique"):
        AppConfig.model_validate(payload)
    payload["features"] = [{"name": "bad", "kind": "eval"}]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_dates_and_prices(market):
    derivatives, benchmark = market
    with pytest.raises(DataError, match="calendars"):
        align_market(derivatives, benchmark.drop(benchmark.index[10]))
    with pytest.raises(DataError, match="one row"):
        validate_market(pd.concat([benchmark, benchmark.iloc[:1]]), ("close",))
    benchmark.iloc[0, 0] = 0
    with pytest.raises(DataError, match="positive"):
        validate_market(benchmark, ("close",))


def test_no_implicit_pickle_loading(tmp_path):
    path = tmp_path / "frame.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(DataError, match="trusted"):
        read_table(path, allow_pickle=False)
    with pytest.raises(DataError, match="not found"):
        read_table(Path("/missing/data.csv"), allow_pickle=False)


def test_vendor_multiindex(market):
    data, _ = market
    data["order_book_id"] = "IF"
    vendor = data.reset_index().set_index(["order_book_id", "date"])
    normalized = validate_market(vendor, ("close",))
    assert normalized.index.equals(data.index)
