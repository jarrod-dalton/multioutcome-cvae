"""Neutral recovery evidence for mixed-cardinality categorical outcomes.

The acceptance criterion deliberately compares nominal contingency tables,
not covariances of the integer codes.  A covariance would change when a user
reorders otherwise identical category labels.
"""

import numpy as np
import torch

from multioutcome_cvae import CVAETrainer
from multioutcome_cvae.examples.categorical_basic import (
    OUTCOME_SCHEMA,
    simulate_mixed_categorical_data,
)


SEEDS = (1701, 99, 31415)
OUTCOME_PAIRS = ((0, 1), (0, 2), (1, 2))


def _as_rows(values: np.ndarray) -> np.ndarray:
    """Flatten optional within-X draws while retaining semantic outcomes."""
    return np.asarray(values).reshape(-1, len(OUTCOME_SCHEMA))


def _marginal_frequencies(values: np.ndarray, outcome_index: int) -> np.ndarray:
    rows = _as_rows(values)
    cardinality = len(OUTCOME_SCHEMA[outcome_index]["levels"])
    return (
        np.bincount(rows[:, outcome_index], minlength=cardinality).astype(np.float64)
        / rows.shape[0]
    )


def _maximum_marginal_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    _marginal_frequencies(reference, outcome_index)
                    - _marginal_frequencies(candidate, outcome_index)
                )
            )
        )
        for outcome_index in range(len(OUTCOME_SCHEMA))
    )


def _pairwise_contingency_errors(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return total-variation errors between normalized joint tables."""
    reference_rows = _as_rows(reference)
    candidate_rows = _as_rows(candidate)
    errors = []

    for first, second in OUTCOME_PAIRS:
        first_cardinality = len(OUTCOME_SCHEMA[first]["levels"])
        second_cardinality = len(OUTCOME_SCHEMA[second]["levels"])
        table_size = first_cardinality * second_cardinality

        # Integer codes are used only as lossless table-cell indices.  Total
        # variation sums over all cells and is invariant to level reordering.
        reference_cells = (
            reference_rows[:, first] * second_cardinality + reference_rows[:, second]
        )
        candidate_cells = (
            candidate_rows[:, first] * second_cardinality + candidate_rows[:, second]
        )
        reference_table = (
            np.bincount(reference_cells, minlength=table_size).astype(np.float64)
            / reference_rows.shape[0]
        )
        candidate_table = (
            np.bincount(candidate_cells, minlength=table_size).astype(np.float64)
            / candidate_rows.shape[0]
        )
        errors.append(0.5 * np.abs(reference_table - candidate_table).sum())

    return np.asarray(errors)


def _sample_independent_marginals(
    probabilities: dict[str, np.ndarray],
    n_samples_per_x: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the fitted marginals independently, conditional on each X."""
    n_rows = next(iter(probabilities.values())).shape[0]
    samples = np.empty((n_rows, n_samples_per_x, len(OUTCOME_SCHEMA)), dtype=np.int32)

    for outcome_index, schema_entry in enumerate(OUTCOME_SCHEMA):
        outcome_probabilities = probabilities[schema_entry["name"]]
        cumulative = np.cumsum(outcome_probabilities, axis=1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random((n_rows, n_samples_per_x))
        samples[:, :, outcome_index] = np.sum(
            uniforms[:, :, None] > cumulative[:, None, :], axis=2
        ).astype(np.int32)

    return samples


def test_mixed_categorical_recovery_beats_independent_marginals():
    """Recover marginals and improve nominal joint tables over independence."""
    n_samples_per_x = 10
    all_cvae_joint_errors = []
    all_independent_joint_errors = []
    maximum_marginal_error = 0.0

    for seed in SEEDS:
        X, Y = simulate_mixed_categorical_data(n_samples=8000, n_features=4, seed=seed)
        X_train, X_test = X[:6000], X[6000:]
        Y_train, Y_test = Y[:6000], Y[6000:]

        # fit(seed=...) controls training draws; initialization occurs when the
        # trainer is constructed and therefore receives its own explicit seed.
        torch.manual_seed(seed)
        trainer = CVAETrainer(
            x_dim=X.shape[1],
            y_dim=Y.shape[1],
            latent_dim=4,
            outcome_type="categorical",
            outcome_schema=OUTCOME_SCHEMA,
            hidden_dim=64,
            n_hidden_layers=2,
            num_epochs=40,
            batch_size=256,
            device="cpu",
        )
        trainer.fit(
            X_train,
            Y_train,
            X_val=X_test,
            Y_val=Y_test,
            kl_warmup_epochs=10,
            early_stopping_patience=8,
            seed=seed,
            verbose=False,
        )

        torch.manual_seed(seed + 100_000)
        cvae_samples = trainer.generate(
            X_test,
            n_samples_per_x=n_samples_per_x,
            decoder_batch_size=2000,
        )
        assert cvae_samples.dtype == np.int32
        for outcome_index, schema_entry in enumerate(OUTCOME_SCHEMA):
            cardinality = len(schema_entry["levels"])
            assert np.all(cvae_samples[:, :, outcome_index] >= 0)
            assert np.all(cvae_samples[:, :, outcome_index] < cardinality)

        maximum_marginal_error = max(
            maximum_marginal_error,
            _maximum_marginal_error(Y_test, cvae_samples),
        )

        # This control uses the CVAE's own fitted P(Y_j | X), so differences
        # in marginal model quality do not give the joint CVAE an easy win.  It
        # removes only the shared latent draw across semantic outcomes.
        torch.manual_seed(seed + 200_000)
        fitted_marginals = trainer.predict_params(
            X_test, n_mc=100, inference_batch_size=1000
        )["probabilities"]
        independent_samples = _sample_independent_marginals(
            fitted_marginals,
            n_samples_per_x,
            np.random.default_rng(seed + 300_000),
        )

        all_cvae_joint_errors.extend(_pairwise_contingency_errors(Y_test, cvae_samples))
        all_independent_joint_errors.extend(
            _pairwise_contingency_errors(Y_test, independent_samples)
        )

    assert maximum_marginal_error <= 0.05, maximum_marginal_error
    cvae_median_error = float(np.median(all_cvae_joint_errors))
    independent_median_error = float(np.median(all_independent_joint_errors))
    assert cvae_median_error < independent_median_error, (
        cvae_median_error,
        independent_median_error,
    )
