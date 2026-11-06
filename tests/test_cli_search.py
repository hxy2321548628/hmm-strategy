import json

import pandas as pd

from benke.cli import main
from benke.config.settings import AppConfig
from benke.reporting.export import create_run_directory, save_research
from benke.services.research import ResearchService
from benke.services.search import run_search


def test_export_and_search(config, market, monkeypatch):
    data, benchmark = market
    data.to_csv(config.data.derivatives)
    benchmark.to_csv(config.data.benchmark)
    result = ResearchService(config).run()
    directory = create_run_directory(config, "test")
    save_research(result, config, directory)
    assert json.loads((directory / "metrics.json").read_text())["rebalance_count"] >= 0
    assert len(pd.read_csv(directory / "daily.csv")) == len(data) - config.window.train_size
    payload = config.model_dump()
    payload["search"].update(thresholds=[1], train_sizes=[60, 500], test_sizes=[20])
    # Environment must not override a resolved experiment config in workers.
    monkeypatch.setenv("BENKE_WINDOW__TRAIN_SIZE", "9999")
    search = run_search(AppConfig.model_validate(payload))
    assert search.status.tolist() == ["success", "failed"]
    assert "Not enough" in search.error.iloc[1]


def test_cli_schema_and_missing_file(capsys, tmp_path):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["properties"]["features"]
    assert main(["run", "--config", str(tmp_path / "missing.toml")]) == 1
    assert "Error:" in capsys.readouterr().err
