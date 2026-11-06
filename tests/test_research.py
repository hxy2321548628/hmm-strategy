import numpy as np
import pandas as pd
import pytest

from benke.backtesting.metrics import performance
from benke.backtesting.timing import evaluate_signals
from benke.config.settings import AppConfig, BacktestConfig, SignalConfig
from benke.services.research import ResearchService
from benke.strategies.signals import cash_states


def test_window_coverage_tail_and_no_future_leakage(config, market):
    data, benchmark = market
    result = ResearchService(config).run_frames(data, benchmark)
    assert result.daily.index.equals(data.index[60:])
    assert result.windows.test_samples.tolist() == [20, 20, 20, 5]
    assert result.daily.position.iloc[0] == 0
    np.testing.assert_array_equal(result.daily.position.iloc[1:], result.daily.signal.iloc[:-1])
    changed = data.copy()
    changed.iloc[90:, changed.columns.get_loc("close")] *= 1.5
    other = ResearchService(config).run_frames(changed, benchmark)
    pd.testing.assert_frame_equal(result.daily.iloc[:30], other.daily.iloc[:30])


def test_next_day_returns_cost_and_initial_drawdown():
    dates = pd.bdate_range("2020-01-01", periods=4)
    config = BacktestConfig(commission=0.01, slippage=0)
    result = evaluate_signals(
        pd.Series([1, 0, 1, 1], index=dates), pd.Series([0.5, 0.1, -0.2, 0.2], index=dates), config
    )
    np.testing.assert_allclose(result.strategy_return, [0, 0.089, -0.01, 0.188])
    assert performance(pd.Series([-0.2, 0.0]), config)["max_drawdown"] == pytest.approx(-0.2)


def test_state_labels_use_next_return():
    states = np.array([0, 1, 0, 1])
    returns = pd.Series([0.8, -0.2, 0.3, -0.1])
    excluded = cash_states(states, returns, 3, SignalConfig(cumulative_threshold=1))
    assert excluded == {0, 2}


def test_multifactor_and_bic(config, market):
    payload = config.model_dump()
    payload["features"] = [
        {"name": "r", "kind": "log_return", "window": 3},
        {"name": "vol", "kind": "volatility", "window": 5},
    ]
    payload["preprocessing"]["pca_variance"] = 0.8
    payload["hmm"].update(selection="bic", candidates=[2, 3])
    result = ResearchService(AppConfig.model_validate(payload)).run_frames(*market)
    assert result.windows.n_states.isin([2, 3]).all()
    assert result.windows.n_features.ge(1).all()
