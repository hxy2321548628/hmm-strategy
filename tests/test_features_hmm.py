import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from benke.config.settings import FeatureConfig, PreprocessingConfig
from benke.exceptions import ModelError
from benke.features.builder import build_features
from benke.features.preprocessing import FeaturePipeline
from benke.models.hmm import filter_states


def test_formulas_do_not_mutate_or_read_future(market):
    data, _ = market
    original = data.copy(deep=True)
    features = (
        FeatureConfig(name="ma", kind="moving_average", window=5),
        FeatureConfig(name="ret", kind="log_return", window=3),
    )
    result = build_features(data, features)
    assert result.ma.iloc[4] == pytest.approx(data.close.iloc[:5].mean())
    assert result.ret.iloc[3] == pytest.approx(np.log(data.close.iloc[3] / data.close.iloc[0]))
    pd.testing.assert_frame_equal(original, data)
    changed = data.copy()
    changed.iloc[70:, changed.columns.get_loc("close")] *= 2
    pd.testing.assert_frame_equal(result.iloc[:70], build_features(changed, features).iloc[:70])


def test_pca_at_least_one_component_and_training_only():
    features = (
        FeatureConfig(name="a", kind="log_return"),
        FeatureConfig(name="b", kind="log_return"),
    )
    train = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2})
    pipeline = FeaturePipeline(features, PreprocessingConfig(pca_variance=0.8))
    assert pipeline.fit_transform(train).shape == (20, 1)
    mean = pipeline.scaler.mean_.copy()
    first = pipeline.transform(train.iloc[:1])
    batch = pipeline.transform(pd.concat([train.iloc[:1], train * 100]))
    np.testing.assert_allclose(first[0], batch[0])
    np.testing.assert_array_equal(mean, pipeline.scaler.mean_)


def test_boxcox_fixed_domain_and_constant():
    features = (FeatureConfig(name="a", kind="log_return", transform="box_cox"),)
    pipeline = FeaturePipeline(features, PreprocessingConfig())
    pipeline.fit_transform(pd.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0]}))
    with pytest.raises(ModelError, match="domain"):
        pipeline.transform(pd.DataFrame({"a": [-10.0]}))
    with pytest.raises(ModelError, match="Constant"):
        pipeline.fit_transform(pd.DataFrame({"a": [1.0, 1.0, 1.0]}))


@pytest.mark.parametrize("covariance", ["diag", "full", "tied", "spherical"])
def test_filter_matches_prefix_posteriors(covariance):
    model = GaussianHMM(n_components=2, covariance_type=covariance)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    model.means_ = np.array([[-1.0], [1.0]])
    model.n_features = 1
    shapes = {"diag": (2, 1), "full": (2, 1, 1), "tied": (1, 1), "spherical": (2,)}
    model.covars_ = np.ones(shapes[covariance])
    values = np.array([[-0.5], [0.8], [1.5], [-0.8]])
    actual, posterior = filter_states(model, values)
    expected = [model.score_samples(values[: i + 1])[1][-1].argmax() for i in range(len(values))]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(posterior, model.score_samples(values)[1][-1])
    _, first_posterior = filter_states(model, values[:2])
    tail, last_posterior = filter_states(model, values[2:], first_posterior)
    np.testing.assert_array_equal(tail, actual[2:])
    np.testing.assert_allclose(posterior, last_posterior)
