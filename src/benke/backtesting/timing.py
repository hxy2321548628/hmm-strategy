"""Vectorized index-exposure approximation, with one-session signal lag."""

import numpy as np
import pandas as pd

from benke.config.settings import BacktestConfig


def evaluate_signals(
    signals: pd.Series, benchmark_returns: pd.Series, config: BacktestConfig
) -> pd.DataFrame:
    if not signals.index.equals(benchmark_returns.index) or not signals.isin([0, 1]).all():
        raise ValueError("Signals and returns must share dates; signals must be 0 or 1")
    if not np.isfinite(benchmark_returns).all() or (benchmark_returns <= -1).any():
        raise ValueError("Benchmark returns must be finite and greater than -1")
    frame = pd.DataFrame({"benchmark_return": benchmark_returns, "signal": signals})
    frame["position"] = signals.shift(1, fill_value=0)
    frame["turnover"] = frame.position.diff().fillna(frame.position).abs()
    frame["cost"] = frame.turnover * (config.commission + config.slippage)
    gross = frame.position * frame.benchmark_return
    frame["strategy_return"] = (1 - frame.cost) * (1 + gross) - 1
    frame["benchmark_equity"] = (1 + frame.benchmark_return).cumprod()
    frame["strategy_equity"] = (1 + frame.strategy_return).cumprod()
    frame["value"] = config.initial_cash * frame.strategy_equity
    return frame
