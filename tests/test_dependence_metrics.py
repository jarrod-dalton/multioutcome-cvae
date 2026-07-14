import numpy as np
import pytest

from multioutcome_cvae import bernoulli_dependence_metrics


def test_dependence_metrics_are_zero_for_identical_samples():
    rng = np.random.default_rng(123)
    Y = rng.binomial(1, 0.5, size=(200, 5))
    probabilities = np.full(Y.shape, 0.5)

    metrics = bernoulli_dependence_metrics(Y, Y.copy(), probabilities)

    assert metrics["n_pairs"] == 10
    assert metrics["joint_probability_rmse"] == pytest.approx(0.0)
    assert metrics["residual_covariance_rmse"] == pytest.approx(0.0)


def test_dependence_metrics_detect_lost_shared_dependence():
    rng = np.random.default_rng(456)
    latent = rng.binomial(1, 0.5, size=(2000, 1))
    observed = np.repeat(latent, 4, axis=1)
    independent = rng.binomial(1, 0.5, size=observed.shape)

    metrics = bernoulli_dependence_metrics(
        observed,
        independent,
        np.full(observed.shape, 0.5),
    )

    assert metrics["joint_probability_rmse"] > 0.15
    assert metrics["residual_covariance_rmse"] > 0.15