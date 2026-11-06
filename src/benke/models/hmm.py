"""Gaussian HMM fitting and causal forward filtering using public model parameters."""

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from benke.config.enums import StateSelection
from benke.config.settings import HMMConfig
from benke.exceptions import ModelError

logger = logging.getLogger(__name__)


def fit_hmm(values: np.ndarray, config: HMMConfig) -> GaussianHMM:
    candidates = (
        (config.n_states,) if config.selection == StateSelection.FIXED else config.candidates
    )
    models: list[tuple[float, GaussianHMM]] = []
    for states in candidates:
        if len(values) <= states:
            raise ModelError(f"{len(values)} samples are insufficient for {states} states")
        model = GaussianHMM(
            n_components=states,
            covariance_type=config.covariance.value,
            n_iter=config.max_iterations,
            tol=config.tolerance,
            min_covar=config.min_covariance,
            random_state=config.random_seed,
        )
        try:
            model.fit(values)
            score = model.bic(values)
            if not np.isfinite(score):
                raise ValueError("non-finite BIC")
            # Invalid transition rows can occur when a state is never occupied.
            if not np.allclose(model.transmat_.sum(axis=1), 1):
                raise ValueError("invalid transition probabilities")
            models.append((float(score), model))
            history = list(model.monitor_.history)
            if len(history) > 1 and abs(history[-1] - history[-2]) >= config.tolerance:
                logger.warning("HMM (%s states) ended without meeting tolerance", states)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if config.selection == StateSelection.FIXED:
                raise ModelError(f"HMM fit failed ({states} states): {exc}") from exc
            logger.warning("Skipping invalid %s-state candidate: %s", states, exc)
    if not models:
        raise ModelError("All HMM candidates failed")
    return min(models, key=lambda item: item[0])[1]


def filter_states(
    model: GaussianHMM, values: np.ndarray, posterior: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """P(state_t | x_1..x_t), never smooth using future observations.

    The returned posterior can continue the SAME fitted model on a later block.
    A newly fitted model must start with its own training posterior.
    """
    if len(values) == 0:
        raise ModelError("Cannot filter an empty sequence")
    emissions = np.column_stack(
        [
            multivariate_normal.logpdf(values, mean=mean, cov=cov)
            for mean, cov in zip(model.means_, model.covars_, strict=True)
        ]
    )
    states = np.empty(len(values), dtype=int)
    probability = posterior
    for row, emission in enumerate(emissions):
        prior = model.startprob_ if probability is None else probability @ model.transmat_
        log_prior = np.full_like(prior, -np.inf)
        np.log(prior, out=log_prior, where=prior > 0)
        logits = log_prior + emission
        normalizer = logsumexp(logits)
        if not np.isfinite(normalizer):
            raise ModelError("HMM filtering produced an invalid posterior")
        probability = np.exp(logits - normalizer)
        states[row] = probability.argmax()
    return states, probability
