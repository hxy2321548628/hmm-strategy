"""Validated, immutable application configuration; no import-time I/O."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from benke.config.enums import (
    CovarianceType,
    FeatureKind,
    LogLevel,
    PriceColumn,
    StateSelection,
    ThresholdMode,
    TransformKind,
    WeightingMode,
)

PositiveInt = Annotated[int, Field(gt=0)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DataConfig(ConfigModel):
    derivatives: Path
    benchmark: Path
    weights: Path | None = None
    constituents: Path | None = None
    stocks: Path | None = None
    start: date | None = None
    end: date | None = None
    allow_pickle: bool = False

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        if self.start and self.end and self.start > self.end:
            raise ValueError("data.start must be before data.end")
        return self


class FeatureConfig(ConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: FeatureKind
    window: PositiveInt = 5
    slow_window: PositiveInt = 20
    price: PriceColumn = PriceColumn.CLOSE
    transform: TransformKind = TransformKind.NONE

    @model_validator(mode="after")
    def valid_windows(self) -> Self:
        if self.kind == FeatureKind.MOVING_AVERAGE_SPREAD and self.slow_window <= self.window:
            raise ValueError("slow_window must exceed window")
        if self.kind == FeatureKind.VOLATILITY and self.window < 2:
            raise ValueError("volatility needs window >= 2")
        return self


class PreprocessingConfig(ConfigModel):
    standardize: bool = True
    pca_variance: float | None = Field(default=None, gt=0, le=1)
    boxcox_margin: float = Field(default=1.0, gt=0)
    numerical_epsilon: float = Field(default=1e-10, gt=0)


class HMMConfig(ConfigModel):
    selection: StateSelection = StateSelection.FIXED
    n_states: PositiveInt = 5
    candidates: tuple[PositiveInt, ...] = (3, 4, 5, 6, 7)
    covariance: CovarianceType = CovarianceType.DIAGONAL
    max_iterations: PositiveInt = 300
    tolerance: float = Field(default=0.01, gt=0)
    min_covariance: float = Field(default=0.001, gt=0)
    random_seed: int = Field(default=11, ge=0)

    @model_validator(mode="after")
    def valid_candidates(self) -> Self:
        if self.selection == StateSelection.BIC and not self.candidates:
            raise ValueError("BIC selection requires candidates")
        return self


class WindowConfig(ConfigModel):
    train_size: PositiveInt = 520
    test_size: PositiveInt = 30


class SignalConfig(ConfigModel):
    threshold_mode: ThresholdMode = ThresholdMode.FIXED
    cumulative_threshold: float = Field(default=1.08, gt=0)


class BacktestConfig(ConfigModel):
    initial_cash: float = Field(default=10_000_000, gt=0)
    commission: float = Field(default=0.00015, ge=0, lt=1)
    slippage: float = Field(default=0.0001, ge=0, lt=1)
    trading_days: PositiveInt = 252
    annual_risk_free_rate: float = Field(default=0, gt=-1)
    weighting: WeightingMode = WeightingMode.INDEX
    cash_reserve: float = Field(default=0.05, ge=0, lt=1)
    lot_size: PositiveInt = 100

    @model_validator(mode="after")
    def costs_below_capital(self) -> Self:
        if self.commission + self.slippage >= 1:
            raise ValueError("commission + slippage must be below 1")
        return self


class ReportConfig(ConfigModel):
    directory: Path = Path("outputs")
    plot: bool = False
    figure_size: tuple[float, float] = (12, 7)
    dpi: PositiveInt = 150

    @model_validator(mode="after")
    def valid_size(self) -> Self:
        if min(self.figure_size) <= 0:
            raise ValueError("figure dimensions must be positive")
        return self


class SearchConfig(ConfigModel):
    thresholds: tuple[Annotated[float, Field(gt=0)], ...] = (1.0, 1.08, 1.2)
    train_sizes: tuple[PositiveInt, ...] = (520,)
    test_sizes: tuple[PositiveInt, ...] = (30,)
    pca_variances: tuple[Annotated[float, Field(gt=0, le=1)] | None, ...] = (None,)
    max_workers: PositiveInt = 1

    @model_validator(mode="after")
    def nonempty_grid(self) -> Self:
        if not all((self.thresholds, self.train_sizes, self.test_sizes, self.pca_variances)):
            raise ValueError("search dimensions cannot be empty")
        return self


class AppConfig(ConfigModel):
    data: DataConfig
    features: tuple[FeatureConfig, ...]
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    hmm: HMMConfig = Field(default_factory=HMMConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    log_level: LogLevel = LogLevel.INFO

    @model_validator(mode="after")
    def coherent_settings(self) -> Self:
        names = [feature.name for feature in self.features]
        if not names or len(set(names)) != len(names):
            raise ValueError("features must have unique names and cannot be empty")
        warmup = max(
            f.slow_window
            if f.kind == FeatureKind.MOVING_AVERAGE_SPREAD
            else f.window
            if f.kind not in (FeatureKind.HIGH_LOW, FeatureKind.CANDLE_BODY)
            else 0
            for f in self.features
        )
        states = (
            max(self.hmm.candidates)
            if self.hmm.selection == StateSelection.BIC
            else self.hmm.n_states
        )
        if self.window.train_size - warmup <= states:
            raise ValueError("train_size is too small for feature warmup and HMM states")
        return self


class Settings(AppConfig, BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BENKE_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Environment overrides TOML (provided as init_settings), including nested fields.
        return env_settings, init_settings, file_secret_settings


def load_settings(path: str | Path) -> AppConfig:
    """Resolve all relative paths against the TOML file, independent of cwd."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        settings = Settings(**tomllib.load(stream))
    payload = settings.model_dump(mode="python")
    for name in ("derivatives", "benchmark", "weights", "constituents", "stocks"):
        if (value := payload["data"][name]) is not None:
            payload["data"][name] = _resolve(value, path.parent)
    payload["report"]["directory"] = _resolve(settings.report.directory, path.parent)
    # Validate without re-reading the environment after resolving paths.
    return AppConfig.model_validate(payload)


def _resolve(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()
