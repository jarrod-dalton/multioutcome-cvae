import pytest

from multioutcome_cvae import CVAETrainer, simulate_cvae_data


def test_poisson_model_and_simulator_warn_about_deprecated_status():
    with pytest.warns(FutureWarning, match="not production-supported"):
        CVAETrainer(2, 3, outcome_type="poisson")

    with pytest.warns(FutureWarning, match="not production-supported"):
        simulate_cvae_data(10, 2, 3, outcome_type="poisson", seed=1)


def test_negative_binomial_remains_unavailable():
    with pytest.raises(NotImplementedError, match="experimental"):
        CVAETrainer(2, 3, outcome_type="neg_binomial")