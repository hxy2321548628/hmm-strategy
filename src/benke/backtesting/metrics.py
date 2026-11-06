"""Return-based metrics; drawdown includes the initial capital baseline."""

import numpy as np
import pandas as pd

from benke.config.settings import BacktestConfig


def performance(returns: pd.Series, config: BacktestConfig) -> dict[str, float]:
    if returns.empty or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Returns must be finite, non-empty and greater than -1")
    equity = (1 + returns).cumprod()
    years = len(returns) / config.trading_days
    annual = float(equity.iloc[-1] ** (1 / years) - 1)
    volatility = float(returns.std(ddof=0) * np.sqrt(config.trading_days))
    risk_free_daily = (1 + config.annual_risk_free_rate) ** (1 / config.trading_days) - 1
    excess = float((returns - risk_free_daily).mean() * config.trading_days)
    peak = equity.cummax().clip(lower=1)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": annual,
        "annual_volatility": volatility,
        "sharpe": excess / volatility if volatility > 0 else 0.0,
        "max_drawdown": float((equity / peak - 1).min()),
    }
