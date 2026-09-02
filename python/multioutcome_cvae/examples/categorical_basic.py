"""Neutral mixed-cardinality categorical CVAE example.

This example uses three nominal outcomes with 2, 3, and 5 levels.  A shared
latent vector influences every outcome, so fitting separate classifiers would
discard part of the joint dependence that the CVAE is intended to learn.

Run with::

    python -m multioutcome_cvae.examples.categorical_basic
"""

import numpy as np
import torch

from multioutcome_cvae import CVAETrainer


OUTCOME_SCHEMA = [
    {"name": "switch", "levels": ["off", "on"]},
    {"name": "shape", "levels": ["circle", "square", "triangle"]},
    {
        "name": "color",
        "levels": ["amber", "blue", "coral", "green", "violet"],
    },
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def simulate_mixed_categorical_data(
    n_samples: int = 6000,
    n_features: int = 4,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate neutral 2/3/5-level outcomes with shared latent dependence."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    Z = rng.normal(size=(n_samples, 2)).astype(np.float32)
    Y = np.empty((n_samples, len(OUTCOME_SCHEMA)), dtype=np.int32)

    for outcome_index, spec in enumerate(OUTCOME_SCHEMA):
        cardinality = len(spec["levels"])
        x_weights = rng.normal(
            scale=0.45, size=(n_features, cardinality)
        ).astype(np.float32)
        z_weights = rng.normal(scale=1.1, size=(2, cardinality)).astype(np.float32)
        intercept = np.linspace(-0.35, 0.35, cardinality, dtype=np.float32)
        probabilities = _softmax(X @ x_weights + Z @ z_weights + intercept)
        uniforms = rng.random(n_samples)
        cumulative = np.cumsum(probabilities, axis=1)
        draws = np.sum(uniforms[:, None] > cumulative, axis=1)
        Y[:, outcome_index] = np.minimum(draws, cardinality - 1).astype(np.int32)

    return X, Y


def marginal_frequencies(values: np.ndarray, cardinality: int) -> np.ndarray:
    """Return empirical level frequencies, including levels with zero rows."""
    return np.bincount(values, minlength=cardinality) / values.shape[0]


def main() -> None:
    X, Y = simulate_mixed_categorical_data()
    split = 5000
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # fit(seed=...) controls training draws, while this explicit seed controls
    # initialization because the trainer constructs its network before fit().
    torch.manual_seed(2026)
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=4,
        outcome_type="categorical",
        outcome_schema=OUTCOME_SCHEMA,
        hidden_dim=64,
        n_hidden_layers=2,
        num_epochs=40,
    )
    trainer.fit(
        X_train,
        Y_train,
        X_val=X_test,
        Y_val=Y_test,
        kl_warmup_epochs=10,
        early_stopping_patience=6,
        seed=2026,
        verbose=True,
    )

    probabilities = trainer.predict_params(X_test, n_mc=50)["probabilities"]
    generated = trainer.generate(X_test, n_samples_per_x=1)

    print("\nOutcome probability shapes and marginal-frequency errors:")
    for index, spec in enumerate(OUTCOME_SCHEMA):
        name = spec["name"]
        cardinality = len(spec["levels"])
        observed = marginal_frequencies(Y_test[:, index], cardinality)
        simulated = marginal_frequencies(generated[:, index], cardinality)
        print(
            f"  {name}: probabilities={probabilities[name].shape}, "
            f"max_abs_error={np.max(np.abs(observed - simulated)):.3f}"
        )


if __name__ == "__main__":
    main()
