"""Fit transformations on training samples once, reuse unchanged out of sample."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

from benke.config.enums import TransformKind
from benke.config.settings import FeatureConfig, PreprocessingConfig
from benke.exceptions import ModelError


class FeaturePipeline:
    def __init__(self, features: tuple[FeatureConfig, ...], config: PreprocessingConfig):
        self.features = features
        self.config = config
        self.powers: dict[str, PowerTransformer] = {}
        self.offsets: dict[str, float] = {}
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.n_components: int | None = None
        self.fitted = False

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[[feature.name for feature in self.features]].to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ModelError("Features must be non-empty and finite after warmup")
        return values

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fitted = False
        self.powers.clear()
        self.offsets.clear()
        self.scaler = None
        self.pca = None
        values = self._values(frame).copy()
        for column, feature in enumerate(self.features):
            sample = values[:, column]
            if np.std(sample) <= self.config.numerical_epsilon:
                raise ModelError(f"Constant training feature: {feature.name}")
            if feature.transform == TransformKind.NONE:
                continue
            offset = (
                max(0.0, self.config.boxcox_margin - float(sample.min()))
                if feature.transform == TransformKind.BOX_COX
                else 0.0
            )
            method = "box-cox" if feature.transform == TransformKind.BOX_COX else "yeo-johnson"
            transformer = PowerTransformer(method=method, standardize=False)
            try:
                values[:, column] = transformer.fit_transform((sample + offset)[:, None]).ravel()
            except (ValueError, FloatingPointError) as exc:
                raise ModelError(f"Cannot fit power transform for {feature.name}: {exc}") from exc
            self.powers[feature.name] = transformer
            self.offsets[feature.name] = offset
        if self.config.standardize:
            self.scaler = StandardScaler().fit(values)
            values = self.scaler.transform(values)
        self.n_components = values.shape[1]
        if self.config.pca_variance is not None:
            self.pca = PCA(svd_solver="full").fit(values)
            cumulative = self.pca.explained_variance_ratio_.cumsum()
            self.n_components = min(
                len(cumulative),
                int(np.searchsorted(cumulative, self.config.pca_variance, side="left")) + 1,
            )
            values = self.pca.transform(values)[:, : self.n_components]
        self._check_finite(values)
        self.fitted = True
        return values

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ModelError("FeaturePipeline must be fitted before transform")
        values = self._values(frame).copy()
        for column, feature in enumerate(self.features):
            if feature.name not in self.powers:
                continue
            sample = values[:, column] + self.offsets[feature.name]
            if feature.transform == TransformKind.BOX_COX and (sample <= 0).any():
                raise ModelError(
                    f"{feature.name}: test values outside fitted Box-Cox domain; "
                    "use yeo_johnson for unbounded features"
                )
            values[:, column] = self.powers[feature.name].transform(sample[:, None]).ravel()
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if self.pca is not None:
            values = self.pca.transform(values)[:, : self.n_components]
        self._check_finite(values)
        return values

    @staticmethod
    def _check_finite(values: np.ndarray) -> None:
        if not np.isfinite(values).all():
            raise ModelError("Feature transformation produced non-finite values")
