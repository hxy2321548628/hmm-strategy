"""Stable names shared by configuration, features and strategy code."""

from enum import StrEnum


class FeatureKind(StrEnum):
    MOVING_AVERAGE = "moving_average"
    MOVING_AVERAGE_SPREAD = "moving_average_spread"
    LOG_RETURN = "log_return"
    VOLATILITY = "volatility"
    HIGH_LOW = "high_low"
    CANDLE_BODY = "candle_body"


class PriceColumn(StrEnum):
    CLOSE = "close"
    SETTLEMENT = "settlement"


class TransformKind(StrEnum):
    NONE = "none"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"


class CovarianceType(StrEnum):
    DIAGONAL = "diag"
    FULL = "full"
    TIED = "tied"
    SPHERICAL = "spherical"


class StateSelection(StrEnum):
    FIXED = "fixed"
    BIC = "bic"


class ThresholdMode(StrEnum):
    FIXED = "fixed"
    BENCHMARK = "benchmark"


class WeightingMode(StrEnum):
    EQUAL = "equal"
    INDEX = "index"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
