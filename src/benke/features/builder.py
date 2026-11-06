"""Causal feature formulas. Each builder returns one explicitly named column."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from benke.config.enums import FeatureKind
from benke.config.settings import FeatureConfig
from benke.data.repository import validate_market

FeatureBuilder = Callable[[pd.DataFrame, FeatureConfig], pd.Series]


def moving_average(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    return data[config.price].rolling(config.window).mean()


def moving_average_spread(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    prices = data[config.price]
    return prices.rolling(config.window).mean() - prices.rolling(config.slow_window).mean()


def log_return(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    return np.log(data[config.price]).diff(config.window)


def volatility(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    return np.log(data[config.price]).diff().rolling(config.window).std()


def high_low(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    return data.high - data.low


def candle_body(data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    return (data.close - data.open).abs() * data.close / data.open


BUILDERS: dict[FeatureKind, FeatureBuilder] = {
    FeatureKind.MOVING_AVERAGE: moving_average,
    FeatureKind.MOVING_AVERAGE_SPREAD: moving_average_spread,
    FeatureKind.LOG_RETURN: log_return,
    FeatureKind.VOLATILITY: volatility,
    FeatureKind.HIGH_LOW: high_low,
    FeatureKind.CANDLE_BODY: candle_body,
}


def build_features(data: pd.DataFrame, configs: tuple[FeatureConfig, ...]) -> pd.DataFrame:
    required = set()
    for config in configs:
        if config.kind == FeatureKind.HIGH_LOW:
            required.update(("high", "low"))
        elif config.kind == FeatureKind.CANDLE_BODY:
            required.update(("open", "close"))
        else:
            required.add(config.price.value)
    data = validate_market(data, tuple(sorted(required)))
    return pd.DataFrame({c.name: BUILDERS[c.kind](data, c) for c in configs}, index=data.index)
