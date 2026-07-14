import numpy as np
import pytest

from multioutcome_cvae import CVAETrainer


def _trainer():
    return CVAETrainer(
        x_dim=2,
        y_dim=3,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=8,
        n_hidden_layers=1,
    )


@pytest.mark.parametrize(
    ("X", "Y", "match"),
    [
        (
            np.array([[0.0, np.nan], [1.0, 2.0]]),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            "X must contain only finite values",
        ),
        (
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.array([[0.0, 1.0, 0.0], [1.0, np.inf, 1.0]]),
            "Y_train must contain only finite values",
        ),
        (
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 1.0]]),
            "Bernoulli outcomes must be exactly 0 or 1",
        ),
    ],
)
def test_fit_rejects_invalid_complete_data(X, Y, match):
    with pytest.raises(ValueError, match=match):
        _trainer().fit(X, Y, epochs=1, verbose=False)


def test_standardize_requires_fitted_training_state():
    with pytest.raises(RuntimeError, match="standardizer has not been fitted"):
        _trainer()._standardize(np.zeros((2, 2), dtype=np.float32))


def test_predict_rejects_wrong_width_and_nonfinite_values():
    trainer = _trainer()
    X = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    Y = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    trainer.fit(X, Y, epochs=1, verbose=False, seed=1)

    with pytest.raises(ValueError, match="exactly 2 columns"):
        trainer.predict_proba(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="X must contain only finite values"):
        trainer.predict_proba(np.array([[0.0, np.nan]], dtype=np.float32))