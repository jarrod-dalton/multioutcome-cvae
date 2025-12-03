# tests/test_simulate.py
import numpy as np
from multioutcome_cvae import simulate_cvae_data

def test_simulate_bernoulli_shapes():
    X, Y, params = simulate_cvae_data(
        n_samples=1000,
        n_features=5,
        n_outcomes=10,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=1234,
    )
    assert X.shape == (1000, 5)
    assert Y.shape == (1000, 10)

def test_simulate_bernoulli_means():
    X, Y, params = simulate_cvae_data(
        n_samples=20000,
        n_features=3,
        n_outcomes=4,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=555,
    )
    # Each dimension of Y has different mean p_j
    y_means = Y.mean(axis=0)
    assert np.all((y_means > 0.05) & (y_means < 0.95))  # not degenerate

def test_simulate_gaussian_stats():
    X, Y, params = simulate_cvae_data(
        n_samples=30000,
        n_features=4,
        n_outcomes=3,
        latent_dim=2,
        outcome_type="gaussian",
        seed=999,
    )
    y_means = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    assert np.all(np.abs(y_means) < 5)
    assert np.all(y_std > 0.2)

def test_simulate_poisson_stats():
    X, Y, params = simulate_cvae_data(
        n_samples=30000,
        n_features=3,
        n_outcomes=3,
        latent_dim=1,
        outcome_type="poisson",
        seed=2025,
    )
    means = Y.mean(axis=0)
    assert np.all(means > 0)
    assert np.all(means < 20)
