import numpy as np
import pandas as pd
import pytest

from benke.backtesting.portfolio import run_portfolio, target_weights
from benke.config.settings import BacktestConfig
from benke.exceptions import DataError


def test_weight_snapshot_is_asof_and_normalized():
    dates = pd.bdate_range("2020-01-01", periods=3)
    signals = pd.Series([1, 1, 1], index=dates)
    weights = pd.DataFrame(
        {"sec_code": ["A", "B", "A", "B"], "weight": [1, 3, 3, 1]},
        index=[dates[0], dates[0], dates[2], dates[2]],
    )
    targets = target_weights(signals, weights, BacktestConfig(cash_reserve=0.2))
    assert targets[dates[1]] == {"A": 0.2, "B": pytest.approx(0.6)}
    assert targets[dates[2]]["A"] == pytest.approx(0.6)
    with pytest.raises(DataError, match="snapshot"):
        target_weights(signals, weights.iloc[2:], BacktestConfig())


def test_portfolio_next_open_and_suspension():
    pytest.importorskip("backtrader")
    dates = pd.bdate_range("2020-01-01", periods=6, name="date")
    signals = pd.Series([1, 1, 0, 0, 0, 0], index=dates)
    stocks = pd.DataFrame(
        {"sec_code": "A", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0},
        index=dates,
    )
    stocks.loc[dates[1], "volume"] = 0
    stocks.loc[dates[2] :, ["open", "high", "low", "close"]] *= 1.1
    weights = pd.DataFrame({"sec_code": ["A"], "weight": [1.0]}, index=dates[:1])
    result = run_portfolio(
        signals,
        stocks,
        weights,
        BacktestConfig(initial_cash=1000, commission=0, slippage=0, cash_reserve=0.2, lot_size=1),
    )
    fills = result.orders.query("status == 'Completed'")
    assert (fills.date > dates[0]).all()
    assert dates[1] not in fills.date.values
    assert result.daily.index.equals(dates)
    assert np.isfinite(result.daily.value).all()
    assert not fills.empty


def test_all_cash_does_not_need_stock_feeds():
    dates = pd.bdate_range("2020-01-01", periods=3)
    result = run_portfolio(
        pd.Series(0, index=dates),
        pd.DataFrame(),
        pd.DataFrame(columns=["sec_code", "weight"]),
        BacktestConfig(),
    )
    assert result.daily.strategy_return.eq(0).all()


def test_weights_rebalance_existing_holdings_and_future_listing():
    pytest.importorskip("backtrader")
    dates = pd.bdate_range("2020-01-01", periods=6, name="date")
    signals = pd.Series(1, index=dates)
    a = pd.DataFrame(
        {"sec_code": "A", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1000.0},
        index=dates,
    )
    b = a.iloc[2:].copy()
    b["sec_code"] = "B"
    weights = pd.DataFrame(
        {"sec_code": ["A", "A", "B"], "weight": [1.0, 1.0, 3.0]},
        index=[dates[0], dates[2], dates[2]],
    )
    result = run_portfolio(
        signals,
        pd.concat([a, b]),
        weights,
        BacktestConfig(initial_cash=1000, commission=0, slippage=0, cash_reserve=0.2, lot_size=1),
    )
    fills = result.orders.query("status == 'Completed'")
    assert ((fills.security == "A") & (fills["size"] < 0)).any()
    assert ((fills.security == "B") & (fills["size"] > 0)).any()
    assert np.isfinite(result.daily.value).all()
    assert result.daily.value.iloc[-1] == pytest.approx(1000)
