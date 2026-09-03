"""Phase A predictive validation for categorical CVAEs.

This module deliberately lives outside the installable package.  It defines
fixed, neutral data-generating processes (DGPs), fixed training settings, and
predeclared gates before any model is fitted.  Gate failures are evidence to
report, not exceptions to hide.

Run the complete Phase A protocol from the repository root with::

    python -m validation.categorical_predictive_validation \
        --output docs/categorical_predictive_validation.md
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import multioutcome_cvae as multioutcome_cvae_package
from multioutcome_cvae import CVAETrainer


ONE_SIDED_95_Z = 1.6448536269514722
TWO_SIDED_95_Z = 1.959963984540054

# Exact one-sided 95% Student-t critical values used for independent seed-level
# summaries. The locked Phase A protocol uses df=2; nearby entries keep the
# utility useful for small fixed extensions without adding SciPy.
_ONE_SIDED_T_95 = {
    1: 6.313751515,
    2: 2.919985580,
    3: 2.353363435,
    4: 2.131846786,
    5: 2.015048373,
    6: 1.943180281,
    7: 1.894578605,
    8: 1.859548038,
    9: 1.833112933,
    10: 1.812461123,
}


@dataclass(frozen=True)
class OutcomeDGP:
    """Fixed parameters for one nominal outcome."""

    name: str
    levels: Tuple[str, ...]
    intercepts: Tuple[float, ...]
    x_weights: Tuple[Tuple[float, ...], ...]
    latent_weights: Tuple[Tuple[float, ...], ...]

    def validate(self, x_dim: int, latent_dim: int) -> None:
        cardinality = len(self.levels)
        if cardinality < 2 or len(set(self.levels)) != cardinality:
            raise ValueError(f"{self.name}: levels must contain >=2 unique labels.")
        if len(self.intercepts) != cardinality:
            raise ValueError(f"{self.name}: intercept width is inconsistent.")
        if np.asarray(self.x_weights).shape != (x_dim, cardinality):
            raise ValueError(f"{self.name}: x_weights has an invalid shape.")
        if np.asarray(self.latent_weights).shape != (latent_dim, cardinality):
            raise ValueError(f"{self.name}: latent_weights has an invalid shape.")


@dataclass(frozen=True)
class ScenarioDGP:
    """A fully fixed multivariate categorical DGP."""

    name: str
    description: str
    x_dim: int
    latent_dim: int
    conditionally_independent: bool
    outcomes: Tuple[OutcomeDGP, ...]

    def validate(self) -> None:
        if self.x_dim < 1 or self.latent_dim < 1:
            raise ValueError("DGP dimensions must be positive.")
        names = [outcome.name for outcome in self.outcomes]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{self.name}: outcome names must be unique.")
        for outcome in self.outcomes:
            outcome.validate(self.x_dim, self.latent_dim)
            latent = np.asarray(outcome.latent_weights, dtype=np.float64)
            if self.conditionally_independent and np.any(latent != 0.0):
                raise ValueError(
                    f"{self.name}: conditional-independence DGP has latent effects."
                )

    @property
    def outcome_schema(self) -> List[Dict[str, Any]]:
        return [
            {"name": outcome.name, "levels": list(outcome.levels)}
            for outcome in self.outcomes
        ]


@dataclass(frozen=True)
class DataSplit:
    """One immutable-by-contract role in the validation data."""

    X: np.ndarray
    Y: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class StrictSplits:
    """Disjoint train, validation, and untouched test sets."""

    train: DataSplit
    validation: DataSplit
    test: DataSplit


@dataclass(frozen=True)
class ValidationConfig:
    """Fixed confirmatory protocol settings (no data-driven tuning)."""

    seeds: Tuple[int, ...] = (1701, 9901, 31415)
    n_train: int = 4000
    n_validation: int = 1000
    n_test: int = 5000
    hidden_dim: int = 48
    n_hidden_layers: int = 2
    latent_dim: int = 2
    num_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    beta_kl: float = 0.20
    kl_warmup_epochs: int = 8
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1.0e-4
    early_stopping_start_epoch: int = 8
    fitted_quadrature_order: int = 31
    fitted_quadrature_checks: Tuple[int, ...] = (21, 41)
    oracle_quadrature_order: int = 41
    oracle_x_batch_size: int = 256
    quadrature_test_rows: int = 128
    quadrature_x_batch_size: int = 24
    mc_sizes: Tuple[int, ...] = (16, 64, 256, 1024)
    mc_test_rows: int = 96
    mc_blocks: int = 8
    calibration_bins: int = 10
    probability_epsilon: float = 1.0e-12
    empirical_joint_alpha: float = 0.5

    def validate(self) -> None:
        sizes = (self.n_train, self.n_validation, self.n_test)
        if any(size < 1 for size in sizes):
            raise ValueError("All split sizes must be positive.")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("At least two unique validation seeds are required.")
        if self.latent_dim != 2:
            raise ValueError("This confirmatory protocol fixes latent_dim=2.")
        if tuple(sorted(set(self.mc_sizes))) != self.mc_sizes or self.mc_sizes[0] < 2:
            raise ValueError("mc_sizes must be unique, increasing, and start above 1.")
        if self.mc_sizes[-1] % self.mc_blocks != 0:
            raise ValueError("Largest mc_sizes value must divide evenly into mc_blocks.")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2.")
        if not 1 <= self.early_stopping_start_epoch <= self.num_epochs:
            raise ValueError(
                "early_stopping_start_epoch must be between 1 and num_epochs."
            )
        if self.early_stopping_start_epoch < max(1, self.kl_warmup_epochs):
            raise ValueError(
                "Model selection must not begin before KL warm-up reaches its target."
            )


DEFAULT_CONFIG = ValidationConfig()


# These gates are defined before any outcome is observed.  The runner records
# PASS/FAIL but never raises on an unfavorable result.
PREDECLARED_GATES: Mapping[str, Dict[str, Any]] = {
    "dependent_every_seed_positive": {
        "kind": "predictive",
        "criterion": (
            "minimum seed-level G > 0, where G = NLL(fitted marginal product) "
            "- NLL(shared) = log score(shared) - log score(product)"
        ),
        "threshold": 0.0,
        "direction": "greater",
        "applies_to": "each dependent DGP",
    },
    "dependent_aggregate_lcb": {
        "kind": "predictive",
        "criterion": (
            "one-sided 95% LCB across top-level seeds, after equal-weighting "
            "dependent scenarios within seed, is G > 0.01 nats/vector"
        ),
        "threshold": 0.01,
        "direction": "greater",
        "applies_to": "all dependent DGP seeds",
    },
    "conditional_independence_penalty_ucb": {
        "kind": "predictive",
        "criterion": (
            "one-sided 95% UCB across top-level seeds, after equal-weighting "
            "CI scenarios within seed, is P < 0.01 nats/vector, where P = "
            "-G = NLL(shared) - NLL(fitted marginal product)"
        ),
        "threshold": 0.01,
        "direction": "less",
        "applies_to": "conditional-independence DGP",
    },
    "conditional_independence_each_seed": {
        "kind": "predictive",
        "criterion": "maximum individual-seed conditional-independence penalty P < 0.02",
        "threshold": 0.02,
        "direction": "less",
        "applies_to": "conditional-independence DGP",
    },
    "dependent_oracle_headroom": {
        "kind": "prerequisite",
        "criterion": (
            "minimum seed-level oracle headroom H >= 0.05 nats/vector, where "
            "H = NLL(oracle marginal product) - NLL(oracle shared)"
        ),
        "threshold": 0.05,
        "direction": "greater_or_equal",
        "applies_to": "each dependent DGP",
    },
    "marginal_log_score_lcb": {
        "kind": "predictive",
        "criterion": (
            "one-sided 95% LCB for CVAE minus intercept-null marginal log "
            "score > 0"
        ),
        "threshold": 0.0,
        "direction": "greater",
        "applies_to": "each DGP",
    },
    "marginal_brier_lcb": {
        "kind": "predictive",
        "criterion": (
            "one-sided 95% LCB for intercept-null minus CVAE marginal Brier "
            "score > 0"
        ),
        "threshold": 0.0,
        "direction": "greater",
        "applies_to": "each DGP",
    },
    "quadrature_gain_mean_stability": {
        "kind": "prerequisite",
        "criterion": "absolute change in mean G from GH-31 to GH-41 <= 0.002",
        "threshold": 0.002,
        "direction": "less_or_equal",
        "applies_to": "every scenario seed on the fixed numerical subset",
    },
    "quadrature_p99_stability": {
        "kind": "prerequisite",
        "criterion": (
            "maximum p99 absolute per-row shared-score or G change from "
            "GH-31 to GH-41 <= 0.01"
        ),
        "threshold": 0.01,
        "direction": "less_or_equal",
        "applies_to": "every scenario seed on the fixed numerical subset",
    },
    "label_permutation_invariance": {
        "kind": "prerequisite",
        "criterion": "maximum fixed-probability relabeling discrepancy <= 1e-10",
        "threshold": 1.0e-10,
        "direction": "less_or_equal",
        "applies_to": "CVAE, intercept-null, and oracle probabilities for every seed",
    },
}


def _outcome(
    name: str,
    levels: Sequence[str],
    intercepts: Sequence[float],
    x_weights: Sequence[Sequence[float]],
    latent_weights: Sequence[Sequence[float]],
) -> OutcomeDGP:
    return OutcomeDGP(
        name=name,
        levels=tuple(levels),
        intercepts=tuple(float(value) for value in intercepts),
        x_weights=tuple(tuple(float(value) for value in row) for row in x_weights),
        latent_weights=tuple(
            tuple(float(value) for value in row) for row in latent_weights
        ),
    )


_MIXED_X_PARAMETERS = (
    _outcome(
        "switch",
        ("off", "on"),
        (0.0, 0.15),
        ((0.0, 0.85), (0.0, -0.50), (0.0, 0.35)),
        ((-0.90, 0.90), (-0.45, 0.45)),
    ),
    _outcome(
        "shape",
        ("circle", "square", "triangle"),
        (0.20, -0.10, 0.0),
        ((0.55, -0.35, 0.10), (-0.45, 0.25, 0.60), (0.20, 0.45, -0.50)),
        ((-1.00, 0.10, 0.90), (-0.50, 0.80, -0.30)),
    ),
    _outcome(
        "color",
        ("amber", "blue", "coral", "green", "violet"),
        (0.10, -0.10, 0.15, -0.05, 0.0),
        (
            (0.55, -0.25, 0.20, -0.40, 0.10),
            (-0.30, 0.50, -0.20, 0.35, -0.10),
            (0.15, -0.45, 0.55, -0.10, 0.25),
        ),
        ((-1.30, -0.60, 0.00, 0.60, 1.30), (0.70, -0.80, 0.30, 0.80, -0.70)),
    ),
)


def _without_latent_effects(outcome: OutcomeDGP) -> OutcomeDGP:
    return _outcome(
        outcome.name,
        outcome.levels,
        outcome.intercepts,
        outcome.x_weights,
        np.zeros((2, len(outcome.levels))),
    )


_ALL_BINARY_PARAMETERS = (
    _outcome(
        "signal_a",
        ("0", "1"),
        (0.0, 0.10),
        ((0.0, 0.70), (0.0, -0.45), (0.0, 0.25)),
        ((-1.00, 1.00), (-0.45, 0.45)),
    ),
    _outcome(
        "signal_b",
        ("0", "1"),
        (0.0, -0.15),
        ((0.0, -0.55), (0.0, 0.65), (0.0, 0.35)),
        ((-0.85, 0.85), (0.65, -0.65)),
    ),
    _outcome(
        "signal_c",
        ("0", "1"),
        (0.0, 0.05),
        ((0.0, 0.40), (0.0, 0.30), (0.0, -0.75)),
        ((0.90, -0.90), (-0.70, 0.70)),
    ),
    _outcome(
        "signal_d",
        ("0", "1"),
        (0.0, -0.05),
        ((0.0, -0.35), (0.0, -0.60), (0.0, 0.55)),
        ((-0.70, 0.70), (-0.95, 0.95)),
    ),
    _outcome(
        "signal_e",
        ("0", "1"),
        (0.0, 0.20),
        ((0.0, 0.50), (0.0, 0.40), (0.0, -0.30)),
        ((0.75, -0.75), (-0.55, 0.55)),
    ),
)


SCENARIOS: Mapping[str, ScenarioDGP] = {
    "mixed_dependent": ScenarioDGP(
        name="mixed_dependent",
        description=(
            "Three 2/3/5-level outcomes with shared two-dimensional latent "
            "causes and observed covariate effects."
        ),
        x_dim=3,
        latent_dim=2,
        conditionally_independent=False,
        outcomes=_MIXED_X_PARAMETERS,
    ),
    "mixed_conditional_independence": ScenarioDGP(
        name="mixed_conditional_independence",
        description=(
            "The same 2/3/5-level marginal regressions with strong shared X "
            "causes but zero latent effects; outcomes are independent given X."
        ),
        x_dim=3,
        latent_dim=2,
        conditionally_independent=True,
        outcomes=tuple(_without_latent_effects(outcome) for outcome in _MIXED_X_PARAMETERS),
    ),
    "all_binary_dependent": ScenarioDGP(
        name="all_binary_dependent",
        description=(
            "A vector of five two-level outcomes with shared two-dimensional "
            "latent causes, exercising the binary special case."
        ),
        x_dim=3,
        latent_dim=2,
        conditionally_independent=False,
        outcomes=_ALL_BINARY_PARAMETERS,
    ),
    "all_binary_conditional_independence": ScenarioDGP(
        name="all_binary_conditional_independence",
        description=(
            "Five two-level outcomes with the same strong shared X causes as "
            "the binary dependent DGP but zero latent effects."
        ),
        x_dim=3,
        latent_dim=2,
        conditionally_independent=True,
        outcomes=tuple(
            _without_latent_effects(outcome) for outcome in _ALL_BINARY_PARAMETERS
        ),
    ),
}

for _scenario in SCENARIOS.values():
    _scenario.validate()
DEFAULT_CONFIG.validate()


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    return _logsumexp(values, axis=axis) - math.log(values.shape[axis])


def tensor_gauss_hermite(
    order: int,
    dimensions: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic nodes/weights for a standard-normal expectation."""
    if order < 2 or dimensions < 1:
        raise ValueError("Quadrature order must be >=2 and dimensions positive.")
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(order)
    node_mesh = np.meshgrid(*([nodes_1d] * dimensions), indexing="ij")
    weight_mesh = np.meshgrid(*([weights_1d] * dimensions), indexing="ij")
    nodes = np.stack([mesh.reshape(-1) for mesh in node_mesh], axis=1) * math.sqrt(2.0)
    weights = np.ones(nodes.shape[0], dtype=np.float64)
    for mesh in weight_mesh:
        weights *= mesh.reshape(-1)
    weights /= math.pi ** (dimensions / 2.0)
    weights /= weights.sum()
    return nodes.astype(np.float64), weights.astype(np.float64)


def integrate_shared_and_product_log_masses(
    selected_outcome_log_probabilities: Sequence[np.ndarray],
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Integrate shared-joint and product-of-integrated-marginal masses.

    Each sequence element has shape ``(n_nodes, n_rows)`` and contains the
    conditional log probability of the observed category for one semantic
    outcome.  The shared result integrates the product at common latent nodes;
    the independent result integrates each outcome and then takes the product.
    """
    if not selected_outcome_log_probabilities:
        raise ValueError("At least one semantic outcome is required.")
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in selected_outcome_log_probabilities
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or any(values.shape != shape for values in arrays):
        raise ValueError("All selected log-probability arrays must share (nodes, rows).")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (shape[0],) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Quadrature weights must be finite, positive, and node-aligned.")
    normalized = weights / weights.sum()
    log_weights = np.log(normalized)[:, None]
    shared = _logsumexp(log_weights + np.sum(np.stack(arrays, axis=2), axis=2), axis=0)
    product = np.sum(
        np.stack(
            [_logsumexp(log_weights + values, axis=0) for values in arrays],
            axis=1,
        ),
        axis=1,
    )
    return {"shared": shared, "product": product, "gain": shared - product}


def dgp_conditional_probabilities(
    scenario: ScenarioDGP,
    X: np.ndarray,
    Z: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Evaluate P(Y_j | X, Z) for all fixed DGP outcome heads."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != scenario.x_dim:
        raise ValueError("X has the wrong shape for the scenario.")
    if Z.ndim != 2 or Z.shape != (X.shape[0], scenario.latent_dim):
        raise ValueError("Z has the wrong shape for the scenario.")
    probabilities = {}
    for outcome in scenario.outcomes:
        logits = (
            np.asarray(outcome.intercepts)[None, :]
            + X @ np.asarray(outcome.x_weights)
            + Z @ np.asarray(outcome.latent_weights)
        )
        probabilities[outcome.name] = _softmax(logits)
    return probabilities


def simulate_strict_splits(
    scenario: ScenarioDGP,
    seed: int,
    config: ValidationConfig = DEFAULT_CONFIG,
) -> StrictSplits:
    """Generate roles from independent deterministic SeedSequence children."""
    scenario.validate()
    config.validate()
    total = config.n_train + config.n_validation + config.n_test
    role_sequences = np.random.SeedSequence(seed).spawn(3)

    def generate_role(
        n_rows: int,
        role_sequence: np.random.SeedSequence,
        row_offset: int,
    ) -> DataSplit:
        rng = np.random.default_rng(role_sequence)
        X = rng.normal(size=(n_rows, scenario.x_dim)).astype(np.float32)
        Z = rng.normal(size=(n_rows, scenario.latent_dim))
        conditional = dgp_conditional_probabilities(scenario, X, Z)
        Y = np.empty((n_rows, len(scenario.outcomes)), dtype=np.int32)
        for outcome_index, outcome in enumerate(scenario.outcomes):
            cumulative = np.cumsum(conditional[outcome.name], axis=1)
            cumulative[:, -1] = 1.0
            Y[:, outcome_index] = np.sum(
                rng.random(n_rows)[:, None] > cumulative,
                axis=1,
            ).astype(np.int32)
        return DataSplit(
            X=X,
            Y=Y,
            row_ids=np.arange(row_offset, row_offset + n_rows, dtype=np.int64),
        )

    splits = StrictSplits(
        train=generate_role(config.n_train, role_sequences[0], 0),
        validation=generate_role(
            config.n_validation, role_sequences[1], config.n_train
        ),
        test=generate_role(
            config.n_test,
            role_sequences[2],
            config.n_train + config.n_validation,
        ),
    )
    assert_strict_splits(splits, total)
    return splits


def assert_strict_splits(splits: StrictSplits, expected_total: int) -> None:
    """Raise on role overlap or an incomplete split partition."""
    role_ids = [set(split.row_ids.tolist()) for split in (splits.train, splits.validation, splits.test)]
    if role_ids[0] & role_ids[1] or role_ids[0] & role_ids[2] or role_ids[1] & role_ids[2]:
        raise RuntimeError("Train/validation/test row IDs overlap.")
    if len(set.union(*role_ids)) != expected_total:
        raise RuntimeError("Train/validation/test rows do not form a complete partition.")


def _dgp_probability_cubes(
    scenario: ScenarioDGP,
    X: np.ndarray,
    nodes: np.ndarray,
) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    cubes = {}
    for outcome in scenario.outcomes:
        base = np.asarray(outcome.intercepts)[None, :] + X @ np.asarray(outcome.x_weights)
        latent_logits = nodes @ np.asarray(outcome.latent_weights)
        logits = base[None, :, :] + latent_logits[:, None, :]
        cubes[outcome.name] = _softmax(logits, axis=2)
    return cubes


def dgp_oracle_probabilities(
    scenario: ScenarioDGP,
    X: np.ndarray,
    order: int = 41,
    x_batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Integrate exact DGP marginal probabilities by tensor Gauss-Hermite."""
    X = np.asarray(X, dtype=np.float64)
    nodes, weights = tensor_gauss_hermite(order, scenario.latent_dim)
    result = {
        outcome.name: np.empty((X.shape[0], len(outcome.levels)), dtype=np.float64)
        for outcome in scenario.outcomes
    }
    for start in range(0, X.shape[0], x_batch_size):
        stop = min(start + x_batch_size, X.shape[0])
        cubes = _dgp_probability_cubes(scenario, X[start:stop], nodes)
        for name, cube in cubes.items():
            result[name][start:stop] = np.tensordot(weights, cube, axes=(0, 0))
    return result


def dgp_oracle_log_mass(
    scenario: ScenarioDGP,
    X: np.ndarray,
    Y: np.ndarray,
    order: int = 41,
    x_batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Return oracle shared-latent and product-of-marginals log masses."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int64)
    nodes, weights = tensor_gauss_hermite(order, scenario.latent_dim)
    shared = np.empty(Y.shape[0], dtype=np.float64)
    product = np.empty(Y.shape[0], dtype=np.float64)
    for start in range(0, Y.shape[0], x_batch_size):
        stop = min(start + x_batch_size, Y.shape[0])
        cubes = _dgp_probability_cubes(scenario, X[start:stop], nodes)
        rows = np.arange(stop - start)
        selected_outcomes = []
        for outcome_index, outcome in enumerate(scenario.outcomes):
            selected = np.clip(
                cubes[outcome.name][:, rows, Y[start:stop, outcome_index]],
                np.finfo(np.float64).tiny,
                1.0,
            )
            selected_outcomes.append(np.log(selected))
        integrated = integrate_shared_and_product_log_masses(
            selected_outcomes, weights
        )
        shared[start:stop] = integrated["shared"]
        product[start:stop] = integrated["product"]
    return {
        "joint": shared,
        "fitted_marginal_product": product,
    }


def _intercept_probabilities(
    Y_train: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    n_rows: int,
    alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    probabilities = {}
    for index, entry in enumerate(schema):
        cardinality = len(entry["levels"])
        counts = np.bincount(Y_train[:, index], minlength=cardinality).astype(np.float64)
        frequency = (counts + alpha) / (Y_train.shape[0] + alpha * cardinality)
        probabilities[entry["name"]] = np.broadcast_to(
            frequency[None, :], (n_rows, cardinality)
        ).copy()
    return probabilities


def _selected_log_probabilities(
    probabilities: Mapping[str, np.ndarray],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    epsilon: float,
) -> np.ndarray:
    selected = np.empty((Y.shape[0], len(schema)), dtype=np.float64)
    rows = np.arange(Y.shape[0])
    for index, entry in enumerate(schema):
        matrix = np.asarray(probabilities[entry["name"]], dtype=np.float64)
        selected[:, index] = np.log(
            np.clip(matrix[rows, Y[:, index]], epsilon, 1.0)
        )
    return selected


def marginal_predictive_metrics(
    probabilities: Mapping[str, np.ndarray],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    n_bins: int = 10,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Compute equal-outcome multiclass NLL, Brier, and classwise calibration."""
    Y = np.asarray(Y, dtype=np.int64)
    selected_logs = _selected_log_probabilities(probabilities, Y, schema, epsilon)
    row_nll = -selected_logs.mean(axis=1)
    row_brier_parts = []
    outcomes = {}

    for outcome_index, entry in enumerate(schema):
        name = entry["name"]
        levels = entry["levels"]
        predicted = np.asarray(probabilities[name], dtype=np.float64)
        if predicted.shape != (Y.shape[0], len(levels)):
            raise ValueError(f"Probability matrix for {name!r} has the wrong shape.")
        if not np.isfinite(predicted).all() or np.any(predicted < 0):
            raise ValueError(f"Probability matrix for {name!r} is invalid.")
        row_sums = predicted.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1.0e-6):
            raise ValueError(f"Probability rows for {name!r} do not sum to one.")

        observed = np.eye(len(levels), dtype=np.float64)[Y[:, outcome_index]]
        row_brier = np.square(predicted - observed).sum(axis=1)
        row_brier_parts.append(row_brier)
        classes = {}
        for class_index, level in enumerate(levels):
            class_probability = predicted[:, class_index]
            class_observed = observed[:, class_index]
            calibration_residual = class_observed - class_probability
            calibration_in_large = float(calibration_residual.mean())
            calibration_standard_error = float(
                calibration_residual.std(ddof=1) / math.sqrt(Y.shape[0])
            )
            observed_count = int(class_observed.sum())
            wilson_lower, wilson_upper = _wilson_interval(
                observed_count, Y.shape[0]
            )
            bin_index = np.minimum(
                (class_probability * n_bins).astype(np.int64), n_bins - 1
            )
            weighted_gap = 0.0
            maximum_gap = 0.0
            nonempty_bins = 0
            reliability_bins = []
            for bin_number in range(n_bins):
                in_bin = bin_index == bin_number
                count = int(in_bin.sum())
                if count == 0:
                    continue
                nonempty_bins += 1
                predicted_mean = float(class_probability[in_bin].mean())
                observed_mean = float(class_observed[in_bin].mean())
                signed_gap = observed_mean - predicted_mean
                absolute_gap = abs(signed_gap)
                bin_residual = class_observed[in_bin] - class_probability[in_bin]
                bin_standard_error = (
                    float(bin_residual.std(ddof=1) / math.sqrt(count))
                    if count > 1
                    else None
                )
                bin_wilson_lower, bin_wilson_upper = _wilson_interval(
                    int(class_observed[in_bin].sum()), count
                )
                weighted_gap += (count / Y.shape[0]) * absolute_gap
                maximum_gap = max(maximum_gap, absolute_gap)
                reliability_bins.append(
                    {
                        "bin": bin_number,
                        "count": count,
                        "mean_predicted": predicted_mean,
                        "observed_fraction": observed_mean,
                        "signed_gap": signed_gap,
                        "gap_standard_error": bin_standard_error,
                        "gap_lower_95": (
                            signed_gap - TWO_SIDED_95_Z * bin_standard_error
                            if bin_standard_error is not None
                            else None
                        ),
                        "gap_upper_95": (
                            signed_gap + TWO_SIDED_95_Z * bin_standard_error
                            if bin_standard_error is not None
                            else None
                        ),
                        "observed_wilson_lower_95": bin_wilson_lower,
                        "observed_wilson_upper_95": bin_wilson_upper,
                    }
                )
            classes[str(level)] = {
                "ece": float(weighted_gap),
                "maximum_gap": float(maximum_gap),
                "nonempty_bins": nonempty_bins,
                "mean_predicted": float(class_probability.mean()),
                "observed_fraction": float(class_observed.mean()),
                "calibration_in_large": calibration_in_large,
                "calibration_in_large_standard_error": calibration_standard_error,
                "calibration_in_large_lower_95": (
                    calibration_in_large
                    - TWO_SIDED_95_Z * calibration_standard_error
                ),
                "calibration_in_large_upper_95": (
                    calibration_in_large
                    + TWO_SIDED_95_Z * calibration_standard_error
                ),
                "observed_wilson_lower_95": wilson_lower,
                "observed_wilson_upper_95": wilson_upper,
                "reliability_bins": reliability_bins,
            }

        outcomes[name] = {
            "nll": float(-selected_logs[:, outcome_index].mean()),
            "brier": float(row_brier.mean()),
            "classwise_ece": float(np.mean([value["ece"] for value in classes.values()])),
            "classwise_maximum_gap": float(
                max(value["maximum_gap"] for value in classes.values())
            ),
            "classes": classes,
        }

    row_brier_average = np.stack(row_brier_parts, axis=1).mean(axis=1)
    return {
        "nll": float(row_nll.mean()),
        "brier": float(row_brier_average.mean()),
        "classwise_ece": float(
            np.mean([value["classwise_ece"] for value in outcomes.values()])
        ),
        "classwise_maximum_gap": float(
            max(value["classwise_maximum_gap"] for value in outcomes.values())
        ),
        "outcomes": outcomes,
        "per_row_nll": row_nll,
        "per_row_brier": row_brier_average,
    }


def _wilson_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials.")
    proportion = successes / trials
    z_squared = TWO_SIDED_95_Z ** 2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        TWO_SIDED_95_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials ** 2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def label_permutation_invariance_check(
    probabilities: Mapping[str, np.ndarray],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    permutations: Optional[Sequence[Sequence[int]]] = None,
    n_bins: int = 10,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Check score invariance after a fixed, lossless relabeling of every outcome."""
    Y = np.asarray(Y, dtype=np.int64)
    if permutations is None:
        permutations = [
            tuple(range(len(entry["levels"]) - 1, -1, -1)) for entry in schema
        ]
    if len(permutations) != len(schema):
        raise ValueError("One label permutation is required per semantic outcome.")

    permuted_y = Y.copy()
    permuted_probabilities = {}
    permuted_schema = []
    normalized_permutations = []
    for outcome_index, (entry, supplied_permutation) in enumerate(
        zip(schema, permutations)
    ):
        cardinality = len(entry["levels"])
        permutation = np.asarray(supplied_permutation, dtype=np.int64)
        if permutation.shape != (cardinality,) or set(permutation.tolist()) != set(
            range(cardinality)
        ):
            raise ValueError(f"Invalid label permutation for {entry['name']!r}.")
        inverse = np.empty(cardinality, dtype=np.int64)
        inverse[permutation] = np.arange(cardinality)
        permuted_y[:, outcome_index] = inverse[Y[:, outcome_index]]
        permuted_probabilities[entry["name"]] = np.asarray(
            probabilities[entry["name"]]
        )[:, permutation]
        permuted_schema.append(
            {
                "name": entry["name"],
                "levels": [entry["levels"][old_index] for old_index in permutation],
            }
        )
        normalized_permutations.append(permutation.tolist())

    original = marginal_predictive_metrics(
        probabilities, Y, schema, n_bins=n_bins, epsilon=epsilon
    )
    permuted = marginal_predictive_metrics(
        permuted_probabilities,
        permuted_y,
        permuted_schema,
        n_bins=n_bins,
        epsilon=epsilon,
    )
    metric_changes = {
        metric: abs(float(original[metric]) - float(permuted[metric]))
        for metric in ("nll", "brier", "classwise_ece", "classwise_maximum_gap")
    }
    original_selected = _selected_log_probabilities(
        probabilities, Y, schema, epsilon
    ).sum(axis=1)
    permuted_selected = _selected_log_probabilities(
        permuted_probabilities, permuted_y, permuted_schema, epsilon
    ).sum(axis=1)
    selected_change = float(np.max(np.abs(original_selected - permuted_selected)))
    maximum = max(list(metric_changes.values()) + [selected_change])
    return {
        "permutations_new_to_old": normalized_permutations,
        "metric_absolute_changes": metric_changes,
        "selected_log_mass_max_absolute_change": selected_change,
        "maximum_absolute_discrepancy": float(maximum),
    }


def paired_score_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """Summarize paired per-row score differences (positive favors candidate)."""
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    if difference.ndim != 1 or difference.size < 2 or not np.isfinite(difference).all():
        raise ValueError("Paired scores must be finite one-dimensional arrays.")
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1) / math.sqrt(difference.size))
    return {
        "n": int(difference.size),
        "mean": mean,
        "standard_error": standard_error,
        "lcb_95_one_sided": mean - ONE_SIDED_95_Z * standard_error,
        "ucb_95_one_sided": mean + ONE_SIDED_95_Z * standard_error,
        "lower_95_two_sided": mean - TWO_SIDED_95_Z * standard_error,
        "upper_95_two_sided": mean + TWO_SIDED_95_Z * standard_error,
    }


def _seed_level_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("At least two seed-level values are required.")
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    mean = float(array.mean())
    degrees_of_freedom = int(array.size - 1)
    critical_value = _ONE_SIDED_T_95.get(
        degrees_of_freedom,
        ONE_SIDED_95_Z if degrees_of_freedom > 30 else _ONE_SIDED_T_95[10],
    )
    return {
        "n_seeds": int(array.size),
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical_value,
        "mean": mean,
        "standard_error": standard_error,
        "lcb_95_one_sided": mean - critical_value * standard_error,
        "ucb_95_one_sided": mean + critical_value * standard_error,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def cvae_quadrature_predictions(
    trainer: CVAETrainer,
    X: np.ndarray,
    Y: np.ndarray,
    order: int = 31,
    x_batch_size: int = 24,
) -> Dict[str, Any]:
    """Deterministically integrate fitted-CVAE predictions over N(0, I).

    Decoder evaluation follows the model's float32 contract, while log-softmax
    and all quadrature accumulation use float64.  One quadrature node is shared
    across every semantic outcome in a joint integrand.
    """
    if trainer.outcome_type != "categorical" or not trainer.trained:
        raise ValueError("A fitted categorical CVAETrainer is required.")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y row counts differ.")
    nodes, weights = tensor_gauss_hermite(order, trainer.latent_dim)
    X_standardized = trainer._standardize(X)
    log_weights = torch.from_numpy(np.log(weights)).to(trainer.device, torch.float64)
    z_nodes = torch.from_numpy(nodes.astype(np.float32)).to(trainer.device)
    n_rows = X.shape[0]
    probabilities = {
        entry["name"]: np.empty((n_rows, len(entry["levels"])), dtype=np.float64)
        for entry in trainer.outcome_schema
    }
    joint_log_mass = np.empty(n_rows, dtype=np.float64)
    product_log_mass = np.empty(n_rows, dtype=np.float64)

    trainer.model.eval()
    with torch.no_grad():
        for start in range(0, n_rows, x_batch_size):
            stop = min(start + x_batch_size, n_rows)
            batch_size = stop - start
            x_tensor = torch.from_numpy(X_standardized[start:stop]).to(trainer.device)
            expanded_x = x_tensor.repeat_interleave(nodes.shape[0], dim=0)
            expanded_z = z_nodes.repeat(batch_size, 1)
            logits = trainer.model.decode(expanded_x, expanded_z)["logits"].reshape(
                batch_size, nodes.shape[0], trainer.encoded_y_dim
            )
            joint_terms = log_weights[None, :].expand(batch_size, -1).clone()
            product = torch.zeros(batch_size, device=trainer.device, dtype=torch.float64)
            for outcome_index, (slice_start, slice_stop) in enumerate(
                trainer.outcome_slices
            ):
                group_log_probabilities = F.log_softmax(
                    logits[:, :, slice_start:slice_stop].to(torch.float64), dim=2
                )
                codes = torch.from_numpy(Y[start:stop, outcome_index]).to(
                    trainer.device, torch.long
                )
                selected = group_log_probabilities.gather(
                    2,
                    codes[:, None, None].expand(-1, nodes.shape[0], 1),
                ).squeeze(2)
                joint_terms += selected
                marginal_log_probabilities = torch.logsumexp(
                    log_weights[None, :, None] + group_log_probabilities,
                    dim=1,
                )
                probabilities[trainer.outcome_schema[outcome_index]["name"]][
                    start:stop
                ] = marginal_log_probabilities.exp().cpu().numpy()
                product += torch.logsumexp(log_weights[None, :] + selected, dim=1)
            joint_log_mass[start:stop] = torch.logsumexp(joint_terms, dim=1).cpu().numpy()
            product_log_mass[start:stop] = product.cpu().numpy()

    return {
        "order": order,
        "probabilities": probabilities,
        "joint_log_mass": joint_log_mass,
        "fitted_marginal_product_log_mass": product_log_mass,
    }


def _jackknife_logmeanexp(
    log_values: np.ndarray,
    n_blocks: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return block-jackknife-corrected log means and estimated bias."""
    n_rows, n_draws = log_values.shape
    if n_draws % n_blocks != 0:
        raise ValueError("Number of draws must divide evenly into jackknife blocks.")
    full = _logmeanexp(log_values, axis=1)
    block_width = n_draws // n_blocks
    leave_one_out = np.empty((n_rows, n_blocks), dtype=np.float64)
    for block in range(n_blocks):
        keep = np.ones(n_draws, dtype=bool)
        keep[block * block_width : (block + 1) * block_width] = False
        leave_one_out[:, block] = _logmeanexp(log_values[:, keep], axis=1)
    bias = (n_blocks - 1.0) * (leave_one_out.mean(axis=1) - full)
    return full - bias, bias


def cvae_nested_mc_diagnostic(
    trainer: CVAETrainer,
    X: np.ndarray,
    Y: np.ndarray,
    reference: Mapping[str, np.ndarray],
    mc_sizes: Sequence[int] = (16, 64, 256, 1024),
    seed: int = 700001,
    n_blocks: int = 8,
) -> Dict[str, Any]:
    """Secondary common-panel MC convergence diagnostic against quadrature."""
    sizes = tuple(int(size) for size in mc_sizes)
    if tuple(sorted(set(sizes))) != sizes or sizes[0] < 2:
        raise ValueError("mc_sizes must be increasing, unique, and greater than one.")
    maximum_draws = sizes[-1]
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    X_standardized = trainer._standardize(X)
    rng = np.random.default_rng(seed)
    latent_panel = rng.normal(
        size=(X.shape[0], maximum_draws, trainer.latent_dim)
    ).astype(np.float32)

    x_tensor = torch.from_numpy(X_standardized).to(trainer.device)
    expanded_x = x_tensor.repeat_interleave(maximum_draws, dim=0)
    expanded_z = torch.from_numpy(latent_panel.reshape(-1, trainer.latent_dim)).to(
        trainer.device
    )
    outcome_log_terms = []
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model.decode(expanded_x, expanded_z)["logits"].reshape(
            X.shape[0], maximum_draws, trainer.encoded_y_dim
        )
        for outcome_index, (start, stop) in enumerate(trainer.outcome_slices):
            group = F.log_softmax(logits[:, :, start:stop].to(torch.float64), dim=2)
            codes = torch.from_numpy(Y[:, outcome_index]).to(trainer.device, torch.long)
            selected = group.gather(
                2, codes[:, None, None].expand(-1, maximum_draws, 1)
            ).squeeze(2)
            outcome_log_terms.append(selected.cpu().numpy())

    joint_terms = np.sum(np.stack(outcome_log_terms, axis=2), axis=2)
    path = []
    block_gain_means = []
    for draws in sizes:
        joint = _logmeanexp(joint_terms[:, :draws], axis=1)
        product = np.sum(
            np.stack(
                [_logmeanexp(values[:, :draws], axis=1) for values in outcome_log_terms],
                axis=1,
            ),
            axis=1,
        )
        path.append(
            {
                "draws": draws,
                "joint_nll": float(-joint.mean()),
                "product_nll": float(-product.mean()),
                "mean_gain": float((joint - product).mean()),
                "joint_rmse_vs_quadrature": float(
                    np.sqrt(np.mean(np.square(joint - reference["joint_log_mass"])))
                ),
                "product_rmse_vs_quadrature": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                product
                                - reference["fitted_marginal_product_log_mass"]
                            )
                        )
                    )
                ),
            }
        )

    block_width = maximum_draws // n_blocks
    for block in range(n_blocks):
        section = slice(block * block_width, (block + 1) * block_width)
        block_joint = _logmeanexp(joint_terms[:, section], axis=1)
        block_product = np.sum(
            np.stack(
                [_logmeanexp(values[:, section], axis=1) for values in outcome_log_terms],
                axis=1,
            ),
            axis=1,
        )
        block_gain_means.append(float((block_joint - block_product).mean()))

    corrected_joint, joint_bias = _jackknife_logmeanexp(joint_terms, n_blocks)
    corrected_products = []
    product_biases = []
    for values in outcome_log_terms:
        corrected, bias = _jackknife_logmeanexp(values, n_blocks)
        corrected_products.append(corrected)
        product_biases.append(bias)
    corrected_product = np.stack(corrected_products, axis=1).sum(axis=1)

    shifted = joint_terms - joint_terms.max(axis=1, keepdims=True)
    contribution_weights = np.exp(shifted)
    ess = np.square(contribution_weights.sum(axis=1)) / np.square(
        contribution_weights
    ).sum(axis=1)
    return {
        "seed": seed,
        "path": path,
        "block_gain_means": block_gain_means,
        "block_gain_standard_deviation": float(np.std(block_gain_means, ddof=1)),
        "jackknife_joint_logmass_mean_bias": float(joint_bias.mean()),
        "jackknife_product_logmass_mean_bias": float(
            np.stack(product_biases, axis=1).sum(axis=1).mean()
        ),
        "jackknife_corrected_mean_gain": float(
            (corrected_joint - corrected_product).mean()
        ),
        "joint_integrand_ess_median": float(np.median(ess)),
        "joint_integrand_ess_minimum": float(np.min(ess)),
        "joint_integrand_ess_median_fraction": float(np.median(ess) / maximum_draws),
    }


def _empirical_joint_log_mass(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    alpha: float,
) -> np.ndarray:
    cardinalities = [len(entry["levels"]) for entry in schema]
    multipliers = np.empty(len(cardinalities), dtype=np.int64)
    running_product = 1
    for index in range(len(cardinalities) - 1, -1, -1):
        multipliers[index] = running_product
        running_product *= cardinalities[index]
    train_cells = (Y_train * multipliers[None, :]).sum(axis=1)
    test_cells = (Y_test * multipliers[None, :]).sum(axis=1)
    n_cells = int(np.prod(cardinalities))
    counts = np.bincount(train_cells, minlength=n_cells).astype(np.float64)
    probabilities = (counts + alpha) / (Y_train.shape[0] + alpha * n_cells)
    return np.log(probabilities[test_cells])


def _fit_cvae(
    scenario: ScenarioDGP,
    splits: StrictSplits,
    seed: int,
    config: ValidationConfig,
    device: str,
    verbose: bool,
) -> Tuple[CVAETrainer, Dict[str, Any]]:
    initialization_seed = seed + 100_000
    torch.manual_seed(initialization_seed)
    trainer = CVAETrainer(
        x_dim=scenario.x_dim,
        y_dim=len(scenario.outcomes),
        latent_dim=config.latent_dim,
        outcome_type="categorical",
        outcome_schema=scenario.outcome_schema,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden_layers,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        beta_kl=config.beta_kl,
        device=device,
    )
    history = trainer.fit(
        splits.train.X,
        splits.train.Y,
        X_val=splits.validation.X,
        Y_val=splits.validation.Y,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        beta_kl=config.beta_kl,
        seed=seed + 200_000,
        kl_warmup_epochs=config.kl_warmup_epochs,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_start_epoch=config.early_stopping_start_epoch,
        verbose=verbose,
    )
    return trainer, history


def _strip_row_arrays(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("per_row_nll", "per_row_brier")
    }


def _quadrature_convergence(
    trainer: CVAETrainer,
    X: np.ndarray,
    Y: np.ndarray,
    primary: Mapping[str, Any],
    config: ValidationConfig,
) -> List[Dict[str, float]]:
    rows = min(config.quadrature_test_rows, X.shape[0])
    reference_joint = primary["joint_log_mass"][:rows]
    reference_product = primary["fitted_marginal_product_log_mass"][:rows]
    reference_gain = reference_joint - reference_product
    diagnostics = []
    for order in config.fitted_quadrature_checks:
        check = cvae_quadrature_predictions(
            trainer,
            X[:rows],
            Y[:rows],
            order=order,
            x_batch_size=config.quadrature_x_batch_size,
        )
        check_gain = (
            check["joint_log_mass"]
            - check["fitted_marginal_product_log_mass"]
        )
        joint_absolute_change = np.abs(check["joint_log_mass"] - reference_joint)
        gain_absolute_change = np.abs(check_gain - reference_gain)
        diagnostics.append(
            {
                "order": order,
                "joint_mean_delta_vs_primary": float(
                    (check["joint_log_mass"] - reference_joint).mean()
                ),
                "joint_rmse_vs_primary": float(
                    np.sqrt(
                        np.mean(np.square(check["joint_log_mass"] - reference_joint))
                    )
                ),
                "product_rmse_vs_primary": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                check["fitted_marginal_product_log_mass"]
                                - reference_product
                            )
                        )
                    )
                ),
                "gain_mean_delta_vs_primary": float(
                    check_gain.mean() - reference_gain.mean()
                ),
                "gain_mean_absolute_change": float(
                    abs(check_gain.mean() - reference_gain.mean())
                ),
                "joint_score_absolute_change_p99": float(
                    np.quantile(joint_absolute_change, 0.99)
                ),
                "gain_absolute_change_p99": float(
                    np.quantile(gain_absolute_change, 0.99)
                ),
                "score_or_gain_absolute_change_p99": float(
                    max(
                        np.quantile(joint_absolute_change, 0.99),
                        np.quantile(gain_absolute_change, 0.99),
                    )
                ),
            }
        )
    return diagnostics


def _oracle_ci_mc_identity(
    scenario: ScenarioDGP,
    X: np.ndarray,
    Y: np.ndarray,
    sizes: Sequence[int],
    seed: int,
) -> Dict[int, float]:
    """Exercise the common-panel scorer; CI must be identical at every M."""
    rng = np.random.default_rng(seed)
    maximum = max(sizes)
    Z = rng.normal(size=(X.shape[0] * maximum, scenario.latent_dim))
    repeated_X = np.repeat(X, maximum, axis=0)
    probabilities = dgp_conditional_probabilities(scenario, repeated_X, Z)
    terms = []
    rows = np.arange(X.shape[0] * maximum)
    for outcome_index, outcome in enumerate(scenario.outcomes):
        codes = np.repeat(Y[:, outcome_index], maximum)
        selected = np.log(
            np.clip(probabilities[outcome.name][rows, codes], 1.0e-300, 1.0)
        ).reshape(X.shape[0], maximum)
        terms.append(selected)
    joint_terms = np.stack(terms, axis=2).sum(axis=2)
    result = {}
    for size in sizes:
        shared = _logmeanexp(joint_terms[:, :size], axis=1)
        product = np.stack(
            [_logmeanexp(term[:, :size], axis=1) for term in terms], axis=1
        ).sum(axis=1)
        result[int(size)] = float(np.max(np.abs(shared - product)))
    return result


def evaluate_scenario_seed(
    scenario: ScenarioDGP,
    seed: int,
    config: ValidationConfig = DEFAULT_CONFIG,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit once and evaluate only on the untouched test role."""
    started = time.perf_counter()
    splits = simulate_strict_splits(scenario, seed, config)
    trainer, history = _fit_cvae(scenario, splits, seed, config, device, verbose)
    test = splits.test
    fitted = cvae_quadrature_predictions(
        trainer,
        test.X,
        test.Y,
        order=config.fitted_quadrature_order,
        x_batch_size=config.quadrature_x_batch_size,
    )
    oracle_probabilities = dgp_oracle_probabilities(
        scenario,
        test.X,
        order=config.oracle_quadrature_order,
        x_batch_size=config.oracle_x_batch_size,
    )
    oracle_mass = dgp_oracle_log_mass(
        scenario,
        test.X,
        test.Y,
        order=config.oracle_quadrature_order,
        x_batch_size=config.oracle_x_batch_size,
    )
    intercept_probabilities = _intercept_probabilities(
        splits.train.Y, scenario.outcome_schema, test.Y.shape[0]
    )

    marginal = {
        "cvae": marginal_predictive_metrics(
            fitted["probabilities"],
            test.Y,
            scenario.outcome_schema,
            config.calibration_bins,
            config.probability_epsilon,
        ),
        "intercept_null": marginal_predictive_metrics(
            intercept_probabilities,
            test.Y,
            scenario.outcome_schema,
            config.calibration_bins,
            config.probability_epsilon,
        ),
        "oracle": marginal_predictive_metrics(
            oracle_probabilities,
            test.Y,
            scenario.outcome_schema,
            config.calibration_bins,
            config.probability_epsilon,
        ),
    }
    label_permutation_checks = {
        "cvae": label_permutation_invariance_check(
            fitted["probabilities"],
            test.Y,
            scenario.outcome_schema,
            n_bins=config.calibration_bins,
            epsilon=config.probability_epsilon,
        ),
        "intercept_null": label_permutation_invariance_check(
            intercept_probabilities,
            test.Y,
            scenario.outcome_schema,
            n_bins=config.calibration_bins,
            epsilon=config.probability_epsilon,
        ),
        "oracle": label_permutation_invariance_check(
            oracle_probabilities,
            test.Y,
            scenario.outcome_schema,
            n_bins=config.calibration_bins,
            epsilon=config.probability_epsilon,
        ),
    }

    intercept_joint = _selected_log_probabilities(
        intercept_probabilities,
        test.Y,
        scenario.outcome_schema,
        config.probability_epsilon,
    ).sum(axis=1)
    joint_scores = {
        "cvae": fitted["joint_log_mass"],
        "independent_fitted_marginal": fitted[
            "fitted_marginal_product_log_mass"
        ],
        "intercept_null": intercept_joint,
        "empirical_joint": _empirical_joint_log_mass(
            splits.train.Y,
            test.Y,
            scenario.outcome_schema,
            config.empirical_joint_alpha,
        ),
        "oracle": oracle_mass["joint"],
    }
    joint_summary = {
        name: {
            "nll": float(-values.mean()),
            "nll_standard_error": float(values.std(ddof=1) / math.sqrt(values.size)),
        }
        for name, values in joint_scores.items()
    }
    joint_comparisons = {
        baseline: paired_score_difference(joint_scores["cvae"], values)
        for baseline, values in joint_scores.items()
        if baseline != "cvae"
    }

    cvae_marginal_log_score = -marginal["cvae"]["per_row_nll"]
    intercept_marginal_log_score = -marginal["intercept_null"]["per_row_nll"]
    marginal_log_comparison = paired_score_difference(
        cvae_marginal_log_score, intercept_marginal_log_score
    )
    marginal_brier_improvement = paired_score_difference(
        marginal["intercept_null"]["per_row_brier"],
        marginal["cvae"]["per_row_brier"],
    )

    mc_rows = min(config.mc_test_rows, test.X.shape[0])
    mc_reference = {
        "joint_log_mass": fitted["joint_log_mass"][:mc_rows],
        "fitted_marginal_product_log_mass": fitted[
            "fitted_marginal_product_log_mass"
        ][:mc_rows],
    }
    mc_diagnostic = cvae_nested_mc_diagnostic(
        trainer,
        test.X[:mc_rows],
        test.Y[:mc_rows],
        mc_reference,
        config.mc_sizes,
        seed=seed + 700_000,
        n_blocks=config.mc_blocks,
    )
    quadrature_convergence = _quadrature_convergence(
        trainer, test.X, test.Y, fitted, config
    )

    ci_identity = None
    if scenario.conditionally_independent:
        ci_identity = _oracle_ci_mc_identity(
            scenario,
            test.X[:mc_rows],
            test.Y[:mc_rows],
            config.mc_sizes,
            seed + 800_000,
        )

    return {
        "scenario": scenario.name,
        "seed": seed,
        "conditionally_independent": scenario.conditionally_independent,
        "split_integrity": True,
        "split_sizes": {
            "train": splits.train.Y.shape[0],
            "validation": splits.validation.Y.shape[0],
            "test": splits.test.Y.shape[0],
        },
        "training": {
            "epochs_ran": history["epochs_ran"],
            "best_epoch": history["best_epoch"],
            "final_train_loss": float(history["train_loss"][-1]),
            "restored_best_validation_loss": (
                float(history["val_loss"][history["best_epoch"] - 1])
                if history["best_epoch"] is not None
                else None
            ),
            "best_epoch_train_kl_loss": (
                float(history["train_kl_loss"][history["best_epoch"] - 1])
                if history["best_epoch"] is not None
                else None
            ),
            "best_epoch_train_kl_per_latent": (
                [
                    float(value)
                    for value in history["train_kl_per_latent"][
                        history["best_epoch"] - 1
                    ]
                ]
                if history["best_epoch"] is not None
                else None
            ),
            "best_epoch_active_latent_units": (
                int(history["active_latent_units"][history["best_epoch"] - 1])
                if history["best_epoch"] is not None
                else None
            ),
            "best_epoch_effective_beta_kl": (
                float(history["effective_beta_kl"][history["best_epoch"] - 1])
                if history["best_epoch"] is not None
                else None
            ),
        },
        "marginal": {
            name: _strip_row_arrays(metrics) for name, metrics in marginal.items()
        },
        "marginal_log_score_vs_intercept": marginal_log_comparison,
        "marginal_brier_improvement_vs_intercept": marginal_brier_improvement,
        "joint": joint_summary,
        "joint_comparisons": joint_comparisons,
        "cvae_joint_gain": joint_comparisons["independent_fitted_marginal"],
        "oracle_joint_gain": paired_score_difference(
            oracle_mass["joint"], oracle_mass["fitted_marginal_product"]
        ),
        "quadrature_convergence": quadrature_convergence,
        "mc_diagnostic": mc_diagnostic,
        "oracle_ci_common_panel_identity": ci_identity,
        "label_permutation_checks": label_permutation_checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _gate_result(
    gate: str,
    scope: str,
    value: float,
    detail: str,
) -> Dict[str, Any]:
    definition = PREDECLARED_GATES[gate]
    threshold = float(definition["threshold"])
    direction = definition["direction"]
    if direction == "greater":
        passed = value > threshold
        operator = ">"
    elif direction == "greater_or_equal":
        passed = value >= threshold
        operator = ">="
    elif direction == "less":
        passed = value < threshold
        operator = "<"
    elif direction == "less_or_equal":
        passed = value <= threshold
        operator = "<="
    else:
        raise ValueError(f"Unknown gate direction: {direction!r}.")
    return {
        "gate": gate,
        "kind": definition["kind"],
        "scope": scope,
        "value": float(value),
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
        "detail": detail,
    }


def evaluate_predeclared_gates(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the fixed gates without suppressing or raising on failures."""
    gates = []
    by_scenario = {
        name: [result for result in results if result["scenario"] == name]
        for name in SCENARIOS
    }
    dependent_seed_gains: Dict[int, List[float]] = {}

    for name, scenario in SCENARIOS.items():
        scenario_results = by_scenario[name]
        gains = [result["cvae_joint_gain"]["mean"] for result in scenario_results]
        if not scenario.conditionally_independent:
            for result, gain in zip(scenario_results, gains):
                dependent_seed_gains.setdefault(int(result["seed"]), []).append(gain)
            gates.append(
                _gate_result(
                    "dependent_every_seed_positive",
                    name,
                    min(gains),
                    "minimum of the fixed seed-level paired mean gains",
                )
            )
            oracle_headroom = [
                result["oracle_joint_gain"]["mean"] for result in scenario_results
            ]
            gates.append(
                _gate_result(
                    "dependent_oracle_headroom",
                    name,
                    min(oracle_headroom),
                    "minimum realized-test oracle H across fixed seeds",
                )
            )
        else:
            seed_penalties = [-gain for gain in gains]
            gates.append(
                _gate_result(
                    "conditional_independence_each_seed",
                    name,
                    max(seed_penalties),
                    "maximum individual fixed-seed penalty P",
                )
            )

        marginal_log = _seed_level_summary(
            [
                result["marginal_log_score_vs_intercept"]["mean"]
                for result in scenario_results
            ]
        )
        gates.append(
            _gate_result(
                "marginal_log_score_lcb",
                name,
                marginal_log["lcb_95_one_sided"],
                "one-sided LCB computed across fixed seed-level paired means",
            )
        )

        for result in scenario_results:
            order_41 = next(
                item
                for item in result["quadrature_convergence"]
                if item["order"] == 41
            )
            scope = f"{name}, seed={result['seed']}"
            gates.append(
                _gate_result(
                    "quadrature_gain_mean_stability",
                    scope,
                    order_41["gain_mean_absolute_change"],
                    "fixed-subset absolute mean G difference, GH-31 versus GH-41",
                )
            )
            gates.append(
                _gate_result(
                    "quadrature_p99_stability",
                    scope,
                    order_41["score_or_gain_absolute_change_p99"],
                    "maximum of p99 shared-score and G absolute changes",
                )
            )
            gates.append(
                _gate_result(
                    "label_permutation_invariance",
                    scope,
                    max(
                        check["maximum_absolute_discrepancy"]
                        for check in result["label_permutation_checks"].values()
                    ),
                    "maximum across CVAE, intercept-null, and oracle checks",
                )
            )
        marginal_brier = _seed_level_summary(
            [
                result["marginal_brier_improvement_vs_intercept"]["mean"]
                for result in scenario_results
            ]
        )
        gates.append(
            _gate_result(
                "marginal_brier_lcb",
                name,
                marginal_brier["lcb_95_one_sided"],
                "one-sided LCB computed across fixed seed-level paired means",
            )
        )

    # Scenarios with the same top-level seed deliberately share their X draws.
    # Average scenario contrasts within seed before the t summary so the two
    # scenario rows are not incorrectly treated as independent replications.
    dependent_cluster_means = [
        float(np.mean(dependent_seed_gains[seed]))
        for seed in sorted(dependent_seed_gains)
    ]
    dependent_summary = _seed_level_summary(dependent_cluster_means)
    gates.append(
        _gate_result(
            "dependent_aggregate_lcb",
            "all dependent DGP seeds",
            dependent_summary["lcb_95_one_sided"],
            "one-sided LCB across top-level seeds after averaging dependent scenarios within seed",
        )
    )

    ci_penalties: Dict[int, List[float]] = {}
    for result in results:
        if result["conditionally_independent"]:
            ci_penalties.setdefault(int(result["seed"]), []).append(
                -result["cvae_joint_gain"]["mean"]
            )
    ci_cluster_means = [
        float(np.mean(ci_penalties[seed])) for seed in sorted(ci_penalties)
    ]
    ci_summary = _seed_level_summary(ci_cluster_means)
    gates.append(
        _gate_result(
            "conditional_independence_penalty_ucb",
            "all conditional-independence DGP seeds",
            ci_summary["ucb_95_one_sided"],
            "one-sided UCB across top-level seeds after averaging CI scenarios within seed",
        )
    )
    return gates


def summarize_gate_results(gates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Separate predictive findings from prerequisites and assign a disposition."""

    def counts(selected: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        passed = sum(bool(gate["passed"]) for gate in selected)
        total = len(selected)
        return {"passed": passed, "failed": total - passed, "total": total}

    by_kind = {
        kind: [gate for gate in gates if gate.get("kind") == kind]
        for kind in ("predictive", "prerequisite")
    }
    unknown = [gate for gate in gates if gate.get("kind") not in by_kind]
    if unknown:
        raise ValueError("Every gate result must have a recognized kind.")

    predictive = counts(by_kind["predictive"])
    prerequisite = counts(by_kind["prerequisite"])
    overall = counts(gates)
    if prerequisite["failed"]:
        disposition = "inconclusive"
        display = "INCONCLUSIVE"
        reason = (
            "At least one numerical or design prerequisite failed; predictive "
            "gate outcomes are shown but cannot establish a Phase A pass."
        )
    elif predictive["failed"]:
        disposition = "fail"
        display = "FAIL"
        reason = "All prerequisites passed, but at least one predictive gate failed."
    else:
        disposition = "pass"
        display = "PASS (PHASE A ONLY)"
        reason = (
            "All Phase A prerequisites and predictive gates passed; broader "
            "sensitivity and stress validation remains outside this disposition."
        )
    return {
        "disposition": disposition,
        "display": display,
        "reason": reason,
        "predictive": predictive,
        "prerequisite": prerequisite,
        "all": overall,
    }


def build_protocol_manifest(
    config: ValidationConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Return the exact machine-readable protocol embedded in the report."""
    return {
        "protocol": "categorical-predictive-validation-v1",
        "config": asdict(config),
        "scenarios": {
            name: asdict(scenario) for name, scenario in SCENARIOS.items()
        },
        "gates": dict(PREDECLARED_GATES),
        "primary_estimands": {
            "marginal_log_score": (
                "mean semantic-outcome log predictive probability"
            ),
            "marginal_brier": (
                "mean semantic-outcome multiclass Brier score"
            ),
            "joint_gain_G": (
                "NLL(product of fitted CVAE marginals) - "
                "NLL(shared-latent CVAE joint)"
            ),
            "conditional_independence_penalty_P": "-joint_gain_G",
            "integration": (
                "deterministic tensor Gauss-Hermite under the two-dimensional "
                "standard-normal prior"
            ),
        },
    }


def protocol_manifest_sha256(config: ValidationConfig = DEFAULT_CONFIG) -> str:
    """Hash fixed scenario coefficients and protocol settings deterministically."""
    manifest = build_protocol_manifest(config)
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_environment_metadata() -> Dict[str, str]:
    """Collect package/runtime revision evidence with safe unknown fallbacks."""
    repository_root = Path(__file__).resolve().parents[1]

    def source_metadata(value: Any) -> Tuple[str, str]:
        try:
            source = inspect.getsourcefile(value)
            if source is None:
                return "unknown", "unknown"
            path = Path(source).resolve()
            return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, TypeError):
            return "unknown", "unknown"

    def public_source_location(value: Optional[Path]) -> str:
        if value is None:
            return "unknown"
        try:
            return str(value.relative_to(repository_root))
        except ValueError:
            return "outside repository checkout"

    def git_output(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git",) + arguments,
                cwd=str(repository_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    try:
        package_version = importlib_metadata.version("multioutcome-cvae")
    except importlib_metadata.PackageNotFoundError:
        package_version = "unknown (source tree)"
    package_path = (
        Path(multioutcome_cvae_package.__file__).resolve()
        if multioutcome_cvae_package.__file__ is not None
        else None
    )
    trainer_source_path_raw, trainer_source_sha256 = source_metadata(CVAETrainer)
    validation_source_path_raw, validation_source_sha256 = source_metadata(
        collect_environment_metadata
    )
    trainer_source_path = (
        Path(trainer_source_path_raw) if trainer_source_path_raw != "unknown" else None
    )
    validation_source_path = (
        Path(validation_source_path_raw)
        if validation_source_path_raw != "unknown"
        else None
    )
    expected_trainer_path = (
        repository_root / "python" / "multioutcome_cvae" / "model.py"
    ).resolve()
    dirty_output = git_output("status", "--porcelain")
    git_dirty = "unknown" if dirty_output == "unknown" else str(bool(dirty_output)).lower()
    return {
        "package_version": package_version,
        "package_path": public_source_location(package_path),
        "trainer_source_path": public_source_location(trainer_source_path),
        "trainer_source_sha256": trainer_source_sha256,
        "validation_source_path": public_source_location(validation_source_path),
        "validation_source_sha256": validation_source_sha256,
        "package_source_matches_repository": str(
            trainer_source_path == expected_trainer_path
        ).lower(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_dirty": git_dirty,
    }


def run_validation(
    config: ValidationConfig = DEFAULT_CONFIG,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run all predeclared scenarios and seeds; always retain unfavorable results."""
    config.validate()
    if len(config.seeds) < 3:
        raise ValueError("The full validation run requires at least three seeds.")
    if config.fitted_quadrature_order != 31 or 41 not in config.fitted_quadrature_checks:
        raise ValueError(
            "The predeclared gates require GH-31 as primary and GH-41 as a check."
        )
    environment = collect_environment_metadata()
    if environment["package_source_matches_repository"] != "true":
        raise RuntimeError(
            "The imported CVAETrainer is not the repository source recorded by this "
            "report; install this checkout in editable mode before validation."
        )
    started = time.perf_counter()
    results = []
    for scenario in SCENARIOS.values():
        for seed in config.seeds:
            if verbose:
                print(f"[{scenario.name}] seed={seed}")
            results.append(
                evaluate_scenario_seed(
                    scenario,
                    seed,
                    config=config,
                    device=device,
                    verbose=False,
                )
            )
    gates = evaluate_predeclared_gates(results)
    manifest = build_protocol_manifest(config)
    return {
        "protocol": "categorical-predictive-validation-v1",
        "device": device,
        "environment": environment,
        "protocol_manifest_sha256": protocol_manifest_sha256(config),
        "protocol_manifest": manifest,
        "config": asdict(config),
        "scenarios": results,
        "gates": gates,
        "gate_summary": summarize_gate_results(gates),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(run: Mapping[str, Any]) -> str:
    """Render a complete audit-friendly Markdown report from a validation run."""
    config = run["config"]
    environment = run["environment"]
    manifest_json = json.dumps(
        run["protocol_manifest"], indent=2, sort_keys=True, ensure_ascii=True
    )
    lines = [
        "# Categorical CVAE Phase A predictive validation",
        "",
        f"Protocol: `{run['protocol']}`. Device: `{run['device']}`.",
        "",
        f"Protocol/scenario manifest SHA-256: `{run['protocol_manifest_sha256']}`.",
        "",
        "| environment field | value |",
        "|---|---|",
        f"| package revision | `{_escape_markdown(environment['package_version'])}` |",
        f"| imported package path | `{_escape_markdown(environment['package_path'])}` |",
        f"| trainer source path | `{_escape_markdown(environment['trainer_source_path'])}` |",
        f"| trainer source SHA-256 | `{environment['trainer_source_sha256']}` |",
        f"| validation source path | `{_escape_markdown(environment['validation_source_path'])}` |",
        f"| validation source SHA-256 | `{environment['validation_source_sha256']}` |",
        f"| imported trainer matches checkout | `{environment['package_source_matches_repository']}` |",
        f"| git HEAD | `{_escape_markdown(environment['git_head'])}` |",
        f"| git worktree dirty | `{environment['git_dirty']}` |",
        f"| Python | `{environment['python']}` |",
        f"| NumPy | `{environment['numpy']}` |",
        f"| PyTorch | `{environment['torch']}` |",
        f"| platform | `{_escape_markdown(environment['platform'])}` |",
        "",
        (
            "The scenario classes, primary score, and core acceptance criteria were "
            "posted before the first pilot fit. Exact coefficients, seeds, and model "
            "settings were fixed in the working source; they first become independently "
            "auditable in the committed manifest embedded below. A one-seed pilot was "
            "retained as an unfavorable result in the issue history. Before this "
            "canonical full run, code review corrected two design mechanics: model "
            "selection begins at the end of KL warm-up, and same-seed scenarios are "
            "clustered before seed-level t bounds. Neither correction changed a DGP "
            "strength or a predictive acceptance threshold. Gate failures remain "
            "visible and never abort report generation."
        ),
        "",
        (
            "This is bounded engineering evidence for the four model-aligned core "
            "scenarios. It is not general proof of probabilistic validity, does not "
            "justify a production claim, and does not by itself close GitHub Issue #2."
        ),
        "",
        "## Predeclared protocol",
        "",
        (
            f"Each scenario uses {config['n_train']} training rows, "
            f"{config['n_validation']} validation-only rows, and "
            f"{config['n_test']} untouched test rows. Seeds: "
            f"`{', '.join(str(seed) for seed in config['seeds'])}`."
        ),
        "",
        (
            "Each listed seed determines an independently generated train/validation/"
            "test replicate and separate derived initialization/training streams; no "
            "replicate or initialization is selected after fitting."
        ),
        "",
        (
            "The fitted CVAE primary scores integrate its two-dimensional standard-"
            f"normal prior with {config['fitted_quadrature_order']}x"
            f"{config['fitted_quadrature_order']} tensor Gauss-Hermite nodes. "
            "Monte Carlo estimates are secondary deployment diagnostics only."
        ),
        "",
        "Fixed model settings:",
        "",
        "| latent dim | hidden width | hidden layers | epochs | batch | learning rate | beta KL | warmup | selection starts | patience |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {config['latent_dim']} | {config['hidden_dim']} | "
            f"{config['n_hidden_layers']} | {config['num_epochs']} | "
            f"{config['batch_size']} | {config['learning_rate']} | "
            f"{config['beta_kl']} | {config['kl_warmup_epochs']} | "
            f"{config['early_stopping_start_epoch']} | "
            f"{config['early_stopping_patience']} |"
        ),
        "",
        "Predeclared gates:",
        "",
        "| gate | kind | applies to | criterion |",
        "|---|---|---|---|",
    ]
    for name, definition in PREDECLARED_GATES.items():
        lines.append(
            f"| `{name}` | `{definition['kind']}` | "
            f"{_escape_markdown(definition['applies_to'])} | "
            f"{_escape_markdown(definition['criterion'])} |"
        )

    lines.extend(["", "Scenarios:", "", "| scenario | cardinalities | conditional structure |", "|---|---|---|"])
    for scenario in SCENARIOS.values():
        cardinalities = "/".join(str(len(outcome.levels)) for outcome in scenario.outcomes)
        structure = "independent given X" if scenario.conditionally_independent else "shared latent dependence"
        lines.append(
            f"| `{scenario.name}` | {cardinalities} | {structure}; "
            f"{_escape_markdown(scenario.description)} |"
        )

    lines.extend(
        [
            "",
            "Exact machine-readable protocol manifest:",
            "",
            "```json",
            manifest_json,
            "```",
        ]
    )

    summary = run["gate_summary"]
    lines.extend(
        [
            "",
            "## Gate results",
            "",
            f"Phase A disposition: **{summary['display']}**.",
            "",
            summary["reason"],
            "",
            (
                f"Predictive gates: **{summary['predictive']['passed']} passed / "
                f"{summary['predictive']['failed']} failed / "
                f"{summary['predictive']['total']} total**. Prerequisites: "
                f"**{summary['prerequisite']['passed']} passed / "
                f"{summary['prerequisite']['failed']} failed / "
                f"{summary['prerequisite']['total']} total**."
            ),
            "",
            "| status | kind | gate | scope | value | rule | detail |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for gate in run["gates"]:
        lines.append(
            f"| {'PASS' if gate['passed'] else 'FAIL'} | `{gate['kind']}` | "
            f"`{gate['gate']}` | "
            f"{_escape_markdown(gate['scope'])} | {_format_float(gate['value'], 5)} | "
            f"{gate['operator']} {_format_float(gate['threshold'], 5)} | "
            f"{_escape_markdown(gate['detail'])} |"
        )

    grouped = {
        name: [result for result in run["scenarios"] if result["scenario"] == name]
        for name in SCENARIOS
    }
    for scenario_name, seed_results in grouped.items():
        lines.extend(["", f"## Scenario: `{scenario_name}`", ""])
        lines.extend(
            [
                "### Seed-level primary results",
                "",
                "| seed | epochs (best) | active z at best | KL by latent at best | CVAE marginal NLL | CVAE marginal Brier | joint NLL | fitted-product NLL | paired joint gain (SE) | marginal log gain (SE) | Brier improvement (SE) |",
                "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in seed_results:
            best_kl = " / ".join(
                _format_float(value, 3)
                for value in result["training"]["best_epoch_train_kl_per_latent"]
            )
            lines.append(
                f"| {result['seed']} | {result['training']['epochs_ran']} "
                f"({result['training']['best_epoch']}) | "
                f"{result['training']['best_epoch_active_latent_units']} / "
                f"{config['latent_dim']} | {best_kl} | "
                f"{_format_float(result['marginal']['cvae']['nll'])} | "
                f"{_format_float(result['marginal']['cvae']['brier'])} | "
                f"{_format_float(result['joint']['cvae']['nll'])} | "
                f"{_format_float(result['joint']['independent_fitted_marginal']['nll'])} | "
                f"{_format_float(result['cvae_joint_gain']['mean'])} "
                f"({_format_float(result['cvae_joint_gain']['standard_error'])}) | "
                f"{_format_float(result['marginal_log_score_vs_intercept']['mean'])} "
                f"({_format_float(result['marginal_log_score_vs_intercept']['standard_error'])}) | "
                f"{_format_float(result['marginal_brier_improvement_vs_intercept']['mean'])} "
                f"({_format_float(result['marginal_brier_improvement_vs_intercept']['standard_error'])}) |"
            )

        for result in seed_results:
            lines.extend(["", f"### Seed {result['seed']} detail", ""])
            best_kl = ", ".join(
                _format_float(value, 5)
                for value in result["training"]["best_epoch_train_kl_per_latent"]
            )
            lines.extend(
                [
                    (
                        f"Restored checkpoint: epoch {result['training']['best_epoch']} "
                        f"of {result['training']['epochs_ran']}; validation beta-ELBO "
                        f"{_format_float(result['training']['restored_best_validation_loss'])}; "
                        f"training KL/latent [{best_kl}]; active latent units "
                        f"{result['training']['best_epoch_active_latent_units']} / "
                        f"{config['latent_dim']}; effective beta "
                        f"{_format_float(result['training']['best_epoch_effective_beta_kl'])}."
                    ),
                    "",
                ]
            )
            lines.extend(
                [
                    "Marginal predictive metrics:",
                    "",
                    "| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for model_name, metrics in result["marginal"].items():
                lines.append(
                    f"| `{model_name}` | {_format_float(metrics['nll'])} | "
                    f"{_format_float(metrics['brier'])} | "
                    f"{_format_float(metrics['classwise_ece'])} | "
                    f"{_format_float(metrics['classwise_maximum_gap'])} |"
                )

            lines.extend(
                [
                    "",
                    "Classwise calibration-in-the-large and uncertainty:",
                    "",
                    "| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for model_name, model_metrics in result["marginal"].items():
                for outcome_name, outcome_metrics in model_metrics["outcomes"].items():
                    for level, class_metrics in outcome_metrics["classes"].items():
                        lines.append(
                            f"| `{model_name}` | {_escape_markdown(outcome_name)} | "
                            f"{_escape_markdown(level)} | "
                            f"{_format_float(class_metrics['mean_predicted'])} | "
                            f"{_format_float(class_metrics['observed_fraction'])} | "
                            f"[{_format_float(class_metrics['observed_wilson_lower_95'])}, "
                            f"{_format_float(class_metrics['observed_wilson_upper_95'])}] | "
                            f"{_format_float(class_metrics['calibration_in_large'])} | "
                            f"{_format_float(class_metrics['calibration_in_large_standard_error'])} | "
                            f"[{_format_float(class_metrics['calibration_in_large_lower_95'])}, "
                            f"{_format_float(class_metrics['calibration_in_large_upper_95'])}] | "
                            f"{_format_float(class_metrics['ece'])} | "
                            f"{_format_float(class_metrics['maximum_gap'])} |"
                        )

            lines.extend(
                [
                    "",
                    "CVAE classwise reliability bins (signed gap = observed - predicted):",
                    "",
                    "| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for outcome_name, outcome_metrics in result["marginal"]["cvae"]["outcomes"].items():
                for level, class_metrics in outcome_metrics["classes"].items():
                    for bin_metrics in class_metrics["reliability_bins"]:
                        lines.append(
                            f"| {_escape_markdown(outcome_name)} | {_escape_markdown(level)} | "
                            f"{bin_metrics['bin']} | {bin_metrics['count']} | "
                            f"{_format_float(bin_metrics['mean_predicted'])} | "
                            f"{_format_float(bin_metrics['observed_fraction'])} | "
                            f"[{_format_float(bin_metrics['observed_wilson_lower_95'])}, "
                            f"{_format_float(bin_metrics['observed_wilson_upper_95'])}] | "
                            f"{_format_float(bin_metrics['signed_gap'])} | "
                            f"{_format_float(bin_metrics['gap_standard_error'])} | "
                            f"[{_format_float(bin_metrics['gap_lower_95'])}, "
                            f"{_format_float(bin_metrics['gap_upper_95'])}] |"
                        )

            lines.extend(
                [
                    "",
                    "Fixed label-permutation invariance checks:",
                    "",
                    "| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |",
                    "|---|---:|---:|---:|",
                ]
            )
            for model_name, check in result["label_permutation_checks"].items():
                lines.append(
                    f"| `{model_name}` | "
                    f"{_format_float(max(check['metric_absolute_changes'].values()), 12)} | "
                    f"{_format_float(check['selected_log_mass_max_absolute_change'], 12)} | "
                    f"{_format_float(check['maximum_absolute_discrepancy'], 12)} |"
                )

            lines.extend(
                [
                    "",
                    "Joint predictive scores and paired CVAE comparisons:",
                    "",
                    "| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for model_name, metrics in result["joint"].items():
                comparison = result["joint_comparisons"].get(model_name)
                lines.append(
                    f"| `{model_name}` | {_format_float(metrics['nll'])} | "
                    f"{_format_float(metrics['nll_standard_error'])} | "
                    f"{_format_float(comparison['mean'] if comparison else None)} | "
                    f"{_format_float(comparison['standard_error'] if comparison else None)} | "
                    f"{_format_float(comparison['lcb_95_one_sided'] if comparison else None)} |"
                )

            lines.extend(
                [
                    "",
                    "Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):",
                    "",
                    "| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for item in result["quadrature_convergence"]:
                lines.append(
                    f"| {item['order']} | {_format_float(item['joint_mean_delta_vs_primary'], 6)} | "
                    f"{_format_float(item['joint_rmse_vs_primary'], 6)} | "
                    f"{_format_float(item['product_rmse_vs_primary'], 6)} | "
                    f"{_format_float(item['gain_mean_delta_vs_primary'], 6)} | "
                    f"{_format_float(item['joint_score_absolute_change_p99'], 6)} | "
                    f"{_format_float(item['gain_absolute_change_p99'], 6)} |"
                )

            lines.extend(
                [
                    "",
                    "Secondary nested common-panel Monte Carlo diagnostic:",
                    "",
                    "| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for item in result["mc_diagnostic"]["path"]:
                lines.append(
                    f"| {item['draws']} | {_format_float(item['joint_nll'])} | "
                    f"{_format_float(item['product_nll'])} | {_format_float(item['mean_gain'])} | "
                    f"{_format_float(item['joint_rmse_vs_quadrature'])} | "
                    f"{_format_float(item['product_rmse_vs_quadrature'])} |"
                )
            mc = result["mc_diagnostic"]
            lines.extend(
                [
                    "",
                    (
                        f"MC block gain SD: {_format_float(mc['block_gain_standard_deviation'])}; "
                        f"jackknife-corrected mean gain: "
                        f"{_format_float(mc['jackknife_corrected_mean_gain'])}; median joint-integrand "
                        f"ESS: {_format_float(mc['joint_integrand_ess_median'], 1)} "
                        f"({_format_float(mc['joint_integrand_ess_median_fraction'], 3)} of M)."
                    ),
                ]
            )
            if result["oracle_ci_common_panel_identity"] is not None:
                identity = ", ".join(
                    f"M={size}: {value:.3e}"
                    for size, value in result["oracle_ci_common_panel_identity"].items()
                )
                lines.extend(
                    [
                        "",
                        (
                            "Conditional-independence oracle common-panel shared-vs-product "
                            f"maximum absolute differences: {identity}."
                        ),
                    ]
                )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Joint log predictive mass is the primary dependence metric; integer-code covariance is not used.",
            "- The independent fitted-marginal baseline is the product of the same fitted CVAE marginals, integrated with the same deterministic quadrature rule. This isolates the shared-latent joint contribution.",
            "- The intercept-null and empirical-joint baselines are fitted on training data only. Validation rows are used only for early stopping, and test rows are touched only after fitting.",
            "- Early stopping selects the validation beta-ELBO after warm-up, not held-out prior-predictive log score. Alternative selection criteria are a Phase B sensitivity axis.",
            "- Gauss-Hermite scores, not finite-M log-mean estimates, determine the gates. The nested Monte Carlo panel is diagnostic because finite-M log estimates can have unequal downward bias; M=1 is intentionally excluded.",
            "- Oracle results are a population-optimal reference under the specified DGP, not a finite-sample ceiling: a fitted model can score better on a realized test sample by chance.",
            "- A top-level seed jointly indexes one data replicate and one initialization/training stream. The three-seed bounds measure replicate-level variability but do not separately identify initialization sensitivity.",
            "- Per-row paired standard errors condition on the fitted checkpoint. Only the three-seed summaries incorporate replicate-level data and training variation.",
            "- Fixed-probability label permutation is a scoring and representation sanity check, not a refit-under-relabeling stability experiment; refitted equivalence remains Phase B work.",
            "- The nested Monte Carlo results are descriptive and have no acceptance gate in Phase A.",
            "- Pairwise contingency recovery is covered by the existing synthetic recovery regression test; richer held-out contingency diagnostics and stress distributions remain Phase B work.",
            "- These core DGPs share the fitted model's Gaussian-latent structure. Nonlinear conditional-independence, rare-level, non-Gaussian/Potts, sample-size, latent-capacity, and KL sensitivity arms remain required before Issue #2 can be closed.",
            "- Results apply to the fixed two-dimensional latent configuration only. Other latent dimensions require their own integration and validation evidence.",
            "",
            f"Total runtime: {_format_float(run['runtime_seconds'], 1)} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(run: Mapping[str, Any], output_path: Path) -> Path:
    """Write the complete generated report and return its resolved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(run), encoding="utf-8")
    return path.resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/categorical_predictive_validation.md"),
        help="Path for the generated Markdown report.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device. CPU is the fixed default for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print scenario/seed progress while retaining a concise trainer log.",
    )
    arguments = parser.parse_args(argv)
    run = run_validation(device=arguments.device, verbose=arguments.verbose)
    output = write_markdown_report(run, arguments.output)
    summary = run["gate_summary"]
    print(
        f"Wrote {output} (Phase A disposition: {summary['display']}; "
        f"predictive {summary['predictive']['passed']} passed / "
        f"{summary['predictive']['failed']} failed; prerequisites "
        f"{summary['prerequisite']['passed']} passed / "
        f"{summary['prerequisite']['failed']} failed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
