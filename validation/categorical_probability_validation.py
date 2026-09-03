"""Conditional-probability recovery experiment for categorical CVAEs.

This is the probability-focused companion to
``categorical_predictive_validation.py``.  It asks a narrower question: how
closely does a fitted model recover each known ``P(Y_j = k | X=x)`` under
controlled sample size, prevalence, and cardinality conditions?

The default protocol is intentionally factorial and is not run in ordinary
unit tests.  From the repository root, list or run its fixed scenarios with::

    python -m validation.categorical_probability_validation --list-scenarios
    python -m validation.categorical_probability_validation --output results.json

Outcomes have exact linear-softmax marginal probabilities.  A Gaussian copula
adds dependence without changing those marginals.  Covariates are bounded in
``[-1, 1]`` so rare-category results measure interpolation within a declared
probability envelope rather than uncontrolled Gaussian-tail extrapolation.
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
from validation.categorical_predictive_validation import tensor_gauss_hermite
from validation.probability_diagnostics import compare_probability_models


PROTOCOL_VERSION = "categorical-conditional-probability-v1"
# These seeds are reserved for the canonical run.  Smoke tests use separate
# values and must never fit one of these before the protocol is committed.
DEFAULT_DATA_SEEDS: Tuple[int, ...] = (5003, 10007, 20011, 40009, 80021)
DEFAULT_SAMPLE_SIZES: Tuple[int, ...] = (500, 2000, 8000)
DEFAULT_FOCAL_PREVALENCES: Tuple[float, ...] = (0.01, 0.05, 0.20)
DEFAULT_CARDINALITIES: Tuple[int, ...] = (2, 5, 20)
DEFAULT_HETEROGENEITY_SCHEMAS: Tuple[Tuple[int, ...], ...] = (
    (5, 5, 5, 5),
    (2, 3, 5, 10),
)


def _stable_seed(*parts: Any) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponentiated = np.exp(values[~nonnegative])
    result[~nonnegative] = exponentiated / (1.0 + exponentiated)
    return result


def _canonical_float(value: float) -> float:
    """Remove irrelevant last-bit quadrature/trigonometry platform variation."""
    return float(round(float(value), 12))


def tensor_gauss_legendre_uniform(
    order: int, dimensions: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return tensor Gauss-Legendre nodes/weights for iid Uniform[-1, 1]."""
    if not isinstance(order, int) or order < 2:
        raise ValueError("order must be an integer of at least two.")
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    node_mesh = np.meshgrid(*([nodes_1d] * dimensions), indexing="ij")
    weight_mesh = np.meshgrid(*([weights_1d] * dimensions), indexing="ij")
    nodes = np.stack([item.reshape(-1) for item in node_mesh], axis=1)
    weights = np.ones(nodes.shape[0], dtype=np.float64)
    for item in weight_mesh:
        # Legendre integrates over [-1, 1]; divide each dimension by two to
        # turn the integral into a Uniform[-1, 1] expectation.
        weights *= item.reshape(-1) / 2.0
    weights /= weights.sum()
    return nodes.astype(np.float64), weights.astype(np.float64)


@dataclass(frozen=True)
class ProbabilityStudyConfig:
    """Fixed settings for the full conditional-probability experiment."""

    data_seeds: Tuple[int, ...] = DEFAULT_DATA_SEEDS
    sample_sizes: Tuple[int, ...] = DEFAULT_SAMPLE_SIZES
    validation_sizes: Tuple[Tuple[int, int], ...] = (
        (500, 250),
        (2000, 500),
        (8000, 2000),
    )
    n_test: int = 20_000
    focal_prevalences: Tuple[float, ...] = DEFAULT_FOCAL_PREVALENCES
    homogeneous_cardinalities: Tuple[int, ...] = DEFAULT_CARDINALITIES
    semantic_outcomes: int = 3
    copula_rho: float = 0.35
    include_heterogeneity: bool = True
    heterogeneity_schemas: Tuple[Tuple[int, ...], ...] = (
        DEFAULT_HETEROGENEITY_SCHEMAS
    )
    heterogeneity_anchor_index: int = 2
    heterogeneity_focal_prevalence: float = 0.05
    include_rho_control: bool = True
    rho_control_n_train: int = 2000
    rho_control_prevalence: float = 0.05
    rho_control_cardinality: int = 5
    x_dim: int = 3
    covariate_lower: float = -1.0
    covariate_upper: float = 1.0
    anchor_x_weights: Tuple[float, ...] = (1.35, -0.90, 0.65)
    context_weight_scale: float = 0.90
    population_quadrature_order: int = 15
    latent_dim: int = 2
    hidden_dim: int = 48
    n_hidden_layers: int = 2
    num_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    beta_kl: float = 0.20
    kl_warmup_epochs: int = 8
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1.0e-4
    early_stopping_start_epoch: int = 8
    baseline_l2: float = 1.0e-3
    baseline_max_iter: int = 100
    baseline_tolerance_grad: float = 1.0e-8
    fitted_quadrature_order: int = 21
    fitted_quadrature_check_order: int = 31
    quadrature_check_rows: int = 2000
    quadrature_x_batch_size: int = 128
    calibration_bins: int = 10
    probability_epsilon: float = 1.0e-12
    intercept_alpha: float = 0.5
    plot_rows: int = 1000
    include_initialization_sensitivity: bool = True
    initialization_sensitivity_replicates: int = 3

    def validation_size(self, n_train: int) -> int:
        mapping = dict(self.validation_sizes)
        if n_train not in mapping:
            raise ValueError(f"No validation size was declared for n_train={n_train}.")
        return int(mapping[n_train])

    def validate(self) -> None:
        if not self.data_seeds or len(set(self.data_seeds)) != len(self.data_seeds):
            raise ValueError("data_seeds must contain unique values.")
        if not self.sample_sizes or any(size < 1 for size in self.sample_sizes):
            raise ValueError("sample_sizes must be positive.")
        if set(dict(self.validation_sizes)) != set(self.sample_sizes):
            raise ValueError("validation_sizes must cover sample_sizes exactly.")
        if any(size < 1 for _, size in self.validation_sizes) or self.n_test < 1:
            raise ValueError("Validation and test sizes must be positive.")
        if not self.focal_prevalences or any(
            value <= 0.0 or value >= 1.0 for value in self.focal_prevalences
        ):
            raise ValueError("Focal prevalences must lie strictly between zero and one.")
        if not self.homogeneous_cardinalities or any(
            cardinality < 2 for cardinality in self.homogeneous_cardinalities
        ):
            raise ValueError("Cardinalities must be at least two.")
        if self.semantic_outcomes < 2:
            raise ValueError("At least two semantic outcomes are required.")
        if not 0.0 <= self.copula_rho < 1.0:
            raise ValueError("copula_rho must lie in [0, 1).")
        if self.x_dim != len(self.anchor_x_weights):
            raise ValueError("anchor_x_weights must have x_dim entries.")
        if not (
            math.isfinite(self.covariate_lower)
            and math.isfinite(self.covariate_upper)
            and self.covariate_lower < self.covariate_upper
        ):
            raise ValueError("The covariate bounds are invalid.")
        if self.covariate_lower != -1.0 or self.covariate_upper != 1.0:
            raise ValueError("This locked protocol requires iid Uniform[-1, 1] X.")
        if self.latent_dim != 2:
            raise ValueError("The frozen CVAE uses latent_dim=2.")
        if not 1 <= self.early_stopping_start_epoch <= self.num_epochs:
            raise ValueError("The early-stopping start epoch is invalid.")
        if self.early_stopping_start_epoch < max(1, self.kl_warmup_epochs):
            raise ValueError("Checkpoint selection cannot begin before KL warm-up.")
        if self.baseline_l2 <= 0.0 or self.baseline_max_iter < 1:
            raise ValueError("The independent softmax settings are invalid.")
        if not (
            2 <= self.fitted_quadrature_order < self.fitted_quadrature_check_order
        ):
            raise ValueError("The quadrature check order must exceed the primary order.")
        if self.quadrature_check_rows < 1 or self.quadrature_x_batch_size < 1:
            raise ValueError("Quadrature batch/check sizes must be positive.")
        if self.calibration_bins < 2 or self.plot_rows < 1:
            raise ValueError("Calibration-bin and plot-row counts are invalid.")
        if self.initialization_sensitivity_replicates < 1:
            raise ValueError("initialization_sensitivity_replicates must be positive.")
        if self.include_heterogeneity:
            if len(self.heterogeneity_schemas) != 2:
                raise ValueError("Exactly two matched heterogeneity schemas are required.")
            widths = {len(schema) for schema in self.heterogeneity_schemas}
            totals = {sum(schema) for schema in self.heterogeneity_schemas}
            if len(widths) != 1 or len(totals) != 1:
                raise ValueError(
                    "Heterogeneity schemas must match on outcomes and summed cardinality."
                )
            for schema in self.heterogeneity_schemas:
                if any(cardinality < 2 for cardinality in schema):
                    raise ValueError("Heterogeneity cardinalities must be at least two.")
                if not 0 <= self.heterogeneity_anchor_index < len(schema):
                    raise ValueError("heterogeneity_anchor_index is out of range.")
                if schema[self.heterogeneity_anchor_index] != 5:
                    raise ValueError("The matched heterogeneity anchor must have K=5.")


DEFAULT_CONFIG = ProbabilityStudyConfig()


@dataclass(frozen=True)
class MarginalOutcomeDGP:
    """One exactly specified linear-softmax marginal outcome."""

    name: str
    levels: Tuple[str, ...]
    intercepts: Tuple[float, ...]
    x_weights: Tuple[Tuple[float, ...], ...]
    role: str
    focal_class_index: Optional[int] = None

    @property
    def cardinality(self) -> int:
        return len(self.levels)

    def validate(self, x_dim: int) -> None:
        if not self.name or len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
            raise ValueError("Outcome names and levels must be nonempty and unique.")
        if len(self.intercepts) != self.cardinality:
            raise ValueError(f"{self.name}: intercept width is invalid.")
        if np.asarray(self.x_weights).shape != (x_dim, self.cardinality):
            raise ValueError(f"{self.name}: x_weights shape is invalid.")
        if self.role not in ("anchor", "context"):
            raise ValueError(f"{self.name}: role must be anchor or context.")
        if self.role == "anchor" and self.focal_class_index != 0:
            raise ValueError("The anchor focal class must be zero by contract.")
        if self.role == "context" and self.focal_class_index is not None:
            raise ValueError("Context outcomes cannot define a focal class.")


@dataclass(frozen=True)
class ProbabilityScenario:
    """One fixed cell in the probability-recovery experiment."""

    name: str
    design: str
    variant: str
    pair_key: str
    n_train: int
    n_validation: int
    n_test: int
    focal_prevalence: float
    cardinalities: Tuple[int, ...]
    anchor_index: int
    copula_rho: float
    outcomes: Tuple[MarginalOutcomeDGP, ...]

    @property
    def outcome_schema(self) -> List[Dict[str, Any]]:
        return [
            {"name": outcome.name, "levels": list(outcome.levels)}
            for outcome in self.outcomes
        ]

    @property
    def anchor(self) -> MarginalOutcomeDGP:
        return self.outcomes[self.anchor_index]

    def validate(self, x_dim: int) -> None:
        if self.design not in ("core", "heterogeneity", "rho_control"):
            raise ValueError(f"Unknown design {self.design!r}.")
        if len(self.outcomes) != len(self.cardinalities):
            raise ValueError("Outcome/cardinality widths differ.")
        if tuple(outcome.cardinality for outcome in self.outcomes) != self.cardinalities:
            raise ValueError("Declared cardinalities do not match outcome definitions.")
        if not 0 <= self.anchor_index < len(self.outcomes):
            raise ValueError("anchor_index is out of range.")
        if self.anchor.role != "anchor":
            raise ValueError("anchor_index does not identify the anchor outcome.")
        if sum(outcome.role == "anchor" for outcome in self.outcomes) != 1:
            raise ValueError("Every scenario must have exactly one anchor outcome.")
        if not 0.0 <= self.copula_rho < 1.0:
            raise ValueError("copula_rho must lie in [0, 1).")
        for outcome in self.outcomes:
            outcome.validate(x_dim)


@dataclass(frozen=True)
class ProbabilitySplit:
    X: np.ndarray
    Y: np.ndarray
    row_ids: np.ndarray
    oracle_probabilities: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ProbabilitySplits:
    train: ProbabilitySplit
    validation: ProbabilitySplit
    test: ProbabilitySplit


def _probability_label(value: float) -> str:
    return f"{int(round(1000.0 * value)):03d}"


def _solve_anchor_logit_intercept(
    target: float,
    x_weights: np.ndarray,
    quadrature_order: int,
) -> float:
    """Solve E_X[expit(c + X beta)] = target under Uniform[-1, 1]."""
    nodes, weights = tensor_gauss_legendre_uniform(
        quadrature_order, int(x_weights.size)
    )
    linear = nodes @ x_weights
    lower, upper = -40.0, 40.0
    for _ in range(120):
        midpoint = (lower + upper) / 2.0
        prevalence = float(weights @ _sigmoid(midpoint + linear))
        if prevalence < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _anchor_outcome(
    cardinality: int,
    target_prevalence: float,
    config: ProbabilityStudyConfig,
) -> MarginalOutcomeDGP:
    beta = np.asarray(config.anchor_x_weights, dtype=np.float64)
    binary_intercept = _solve_anchor_logit_intercept(
        target_prevalence, beta, config.population_quadrature_order
    )
    # With K-1 identical zero-logit alternatives, adding log(K-1) makes the
    # focal softmax probability exactly expit(binary_intercept + X beta).
    intercepts = np.zeros(cardinality, dtype=np.float64)
    intercepts[0] = binary_intercept + math.log(cardinality - 1.0)
    x_weights = np.zeros((config.x_dim, cardinality), dtype=np.float64)
    x_weights[:, 0] = beta
    levels = ("focal",) + tuple(
        f"other_{index}" for index in range(1, cardinality)
    )
    return MarginalOutcomeDGP(
        name="anchor",
        levels=levels,
        intercepts=tuple(_canonical_float(value) for value in intercepts),
        x_weights=tuple(
            tuple(_canonical_float(value) for value in row) for row in x_weights
        ),
        role="anchor",
        focal_class_index=0,
    )


def _balanced_context_outcome(
    name: str,
    cardinality: int,
    context_index: int,
    config: ProbabilityStudyConfig,
) -> MarginalOutcomeDGP:
    """Construct an X-predictive context with exactly balanced prevalence."""
    angles = (
        2.0 * math.pi * np.arange(cardinality, dtype=np.float64) / cardinality
        + 0.37 * context_index
    )
    x_weights = np.zeros((config.x_dim, cardinality), dtype=np.float64)
    first = context_index % config.x_dim
    second = (context_index + 1) % config.x_dim
    x_weights[first, :] = config.context_weight_scale * np.cos(angles)
    x_weights[second, :] = config.context_weight_scale * np.sin(angles)

    nodes, weights = tensor_gauss_legendre_uniform(
        config.population_quadrature_order, config.x_dim
    )
    target = np.full(cardinality, 1.0 / cardinality, dtype=np.float64)
    intercepts = np.zeros(cardinality, dtype=np.float64)
    # Multinomial intercept calibration by deterministic iterative scaling.
    # The update converges rapidly for these bounded, moderate coefficients.
    for _ in range(500):
        probabilities = _softmax(intercepts[None, :] + nodes @ x_weights)
        prevalence = weights @ probabilities
        if float(np.max(np.abs(prevalence - target))) <= 1.0e-12:
            break
        intercepts += 0.8 * np.log(target / prevalence)
        intercepts -= intercepts.mean()
    else:  # pragma: no cover - protects future coefficient changes.
        raise RuntimeError("Balanced-context intercept calibration did not converge.")

    return MarginalOutcomeDGP(
        name=name,
        levels=tuple(f"level_{index}" for index in range(cardinality)),
        intercepts=tuple(_canonical_float(value) for value in intercepts),
        x_weights=tuple(
            tuple(_canonical_float(value) for value in row) for row in x_weights
        ),
        role="context",
    )


def _make_scenario(
    *,
    name: str,
    design: str,
    variant: str,
    pair_key: str,
    n_train: int,
    focal_prevalence: float,
    cardinalities: Sequence[int],
    anchor_index: int,
    copula_rho: float,
    config: ProbabilityStudyConfig,
) -> ProbabilityScenario:
    outcomes: List[MarginalOutcomeDGP] = []
    context_number = 0
    for outcome_index, cardinality in enumerate(cardinalities):
        if outcome_index == anchor_index:
            outcomes.append(_anchor_outcome(cardinality, focal_prevalence, config))
        else:
            context_number += 1
            outcomes.append(
                _balanced_context_outcome(
                    f"context_{context_number}",
                    cardinality,
                    context_number,
                    config,
                )
            )
    scenario = ProbabilityScenario(
        name=name,
        design=design,
        variant=variant,
        pair_key=pair_key,
        n_train=n_train,
        n_validation=config.validation_size(n_train),
        n_test=config.n_test,
        focal_prevalence=focal_prevalence,
        cardinalities=tuple(int(value) for value in cardinalities),
        anchor_index=anchor_index,
        copula_rho=copula_rho,
        outcomes=tuple(outcomes),
    )
    scenario.validate(config.x_dim)
    return scenario


def build_probability_scenarios(
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
) -> Tuple[ProbabilityScenario, ...]:
    """Build all fixed core, heterogeneity, and central rho-control cells."""
    config.validate()
    scenarios: List[ProbabilityScenario] = []
    for n_train in config.sample_sizes:
        for prevalence in config.focal_prevalences:
            for cardinality in config.homogeneous_cardinalities:
                name = (
                    f"core_n{n_train}_q{_probability_label(prevalence)}_"
                    f"k{cardinality}"
                )
                scenarios.append(
                    _make_scenario(
                        name=name,
                        design="core",
                        variant="homogeneous",
                        pair_key=name,
                        n_train=n_train,
                        focal_prevalence=prevalence,
                        cardinalities=(cardinality,) * config.semantic_outcomes,
                        anchor_index=0,
                        copula_rho=config.copula_rho,
                        config=config,
                    )
                )

    if config.include_heterogeneity:
        for n_train in config.sample_sizes:
            pair_key = f"heterogeneity_n{n_train}"
            for cardinalities in config.heterogeneity_schemas:
                variant = (
                    "homogeneous"
                    if len(set(cardinalities)) == 1
                    else "heterogeneous"
                )
                name = f"heterogeneity_{variant}_n{n_train}"
                scenarios.append(
                    _make_scenario(
                        name=name,
                        design="heterogeneity",
                        variant=variant,
                        pair_key=pair_key,
                        n_train=n_train,
                        focal_prevalence=config.heterogeneity_focal_prevalence,
                        cardinalities=cardinalities,
                        anchor_index=config.heterogeneity_anchor_index,
                        copula_rho=config.copula_rho,
                        config=config,
                    )
                )

    if config.include_rho_control:
        cardinality = config.rho_control_cardinality
        scenarios.append(
            _make_scenario(
                name=(
                    f"rho_control_n{config.rho_control_n_train}_"
                    f"q{_probability_label(config.rho_control_prevalence)}_"
                    f"k{cardinality}"
                ),
                design="rho_control",
                variant="conditional_independence",
                pair_key=(
                    f"core_n{config.rho_control_n_train}_"
                    f"q{_probability_label(config.rho_control_prevalence)}_"
                    f"k{cardinality}"
                ),
                n_train=config.rho_control_n_train,
                focal_prevalence=config.rho_control_prevalence,
                cardinalities=(cardinality,) * config.semantic_outcomes,
                anchor_index=0,
                copula_rho=0.0,
                config=config,
            )
        )
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise RuntimeError("Generated scenario names are not unique.")
    return tuple(scenarios)


def dgp_conditional_probabilities(
    scenario: ProbabilityScenario, X: np.ndarray
) -> Dict[str, np.ndarray]:
    """Return exact P(Y_j=k | X) matrices from the fixed linear-softmax DGP."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a matrix.")
    x_dim = len(scenario.outcomes[0].x_weights)
    if X.shape[1] != x_dim or not np.isfinite(X).all():
        raise ValueError("X has the wrong width or contains non-finite values.")
    return {
        outcome.name: _softmax(
            np.asarray(outcome.intercepts, dtype=np.float64)[None, :]
            + X @ np.asarray(outcome.x_weights, dtype=np.float64)
        )
        for outcome in scenario.outcomes
    }


def population_mean_probabilities(
    scenario: ProbabilityScenario,
    order: int = 15,
) -> Dict[str, np.ndarray]:
    """Integrate exact DGP marginals over bounded X for protocol checks."""
    x_dim = len(scenario.outcomes[0].x_weights)
    nodes, weights = tensor_gauss_legendre_uniform(order, x_dim)
    probabilities = dgp_conditional_probabilities(scenario, nodes)
    return {name: weights @ matrix for name, matrix in probabilities.items()}


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float64))
    return (
        0.5 * (1.0 + torch.erf(tensor / math.sqrt(2.0)))
    ).numpy()


def _simulate_role(
    scenario: ProbabilityScenario,
    n_rows: int,
    role_seed: np.random.SeedSequence,
    row_offset: int,
    config: ProbabilityStudyConfig,
) -> ProbabilitySplit:
    # Separate children ensure X is unaffected by the number/cardinality of Y
    # draws and make the data-generating roles explicit in the manifest.
    x_sequence, copula_sequence = role_seed.spawn(2)
    x_rng = np.random.default_rng(x_sequence)
    copula_rng = np.random.default_rng(copula_sequence)
    X = x_rng.uniform(
        config.covariate_lower,
        config.covariate_upper,
        size=(n_rows, config.x_dim),
    ).astype(np.float32)
    probabilities = dgp_conditional_probabilities(scenario, X)

    shared = copula_rng.normal(size=n_rows)
    idiosyncratic = copula_rng.normal(size=(n_rows, len(scenario.outcomes)))
    latent_scores = (
        math.sqrt(scenario.copula_rho) * shared[:, None]
        + math.sqrt(1.0 - scenario.copula_rho) * idiosyncratic
    )
    uniforms = np.clip(
        _normal_cdf(latent_scores),
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    Y = np.empty((n_rows, len(scenario.outcomes)), dtype=np.int32)
    for outcome_index, outcome in enumerate(scenario.outcomes):
        cumulative = np.cumsum(probabilities[outcome.name], axis=1)
        cumulative[:, -1] = 1.0
        Y[:, outcome_index] = np.sum(
            uniforms[:, outcome_index, None] > cumulative, axis=1
        ).astype(np.int32)
    return ProbabilitySplit(
        X=X,
        Y=Y,
        row_ids=np.arange(row_offset, row_offset + n_rows, dtype=np.int64),
        oracle_probabilities=probabilities,
    )


def simulate_probability_splits(
    scenario: ProbabilityScenario,
    data_seed: int,
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
) -> ProbabilitySplits:
    """Generate disjoint train/validation/test roles for one scenario replicate."""
    scenario.validate(config.x_dim)
    scenario_key = _stable_seed(PROTOCOL_VERSION, scenario.pair_key)
    role_sequences = np.random.SeedSequence([int(data_seed), scenario_key]).spawn(3)
    train = _simulate_role(
        scenario, scenario.n_train, role_sequences[0], 0, config
    )
    validation = _simulate_role(
        scenario,
        scenario.n_validation,
        role_sequences[1],
        scenario.n_train,
        config,
    )
    test = _simulate_role(
        scenario,
        scenario.n_test,
        role_sequences[2],
        scenario.n_train + scenario.n_validation,
        config,
    )
    roles = (set(train.row_ids), set(validation.row_ids), set(test.row_ids))
    if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
        raise RuntimeError("Train, validation, and test row IDs overlap.")
    if any(
        np.any(split.X < config.covariate_lower)
        or np.any(split.X > config.covariate_upper)
        for split in (train, validation, test)
    ):
        raise RuntimeError("Generated covariates left the declared bounded envelope.")
    return ProbabilitySplits(train=train, validation=validation, test=test)


class IndependentSoftmaxBaseline:
    """Independent ridge multinomial regressions with deterministic fitting.

    Each outcome has its own coefficient matrix.  The heads share neither
    parameters nor latent variables; they are optimized together only to avoid
    repeated framework overhead.  Zero initialization plus full-batch convex
    L-BFGS makes the result independent of a random initialization.
    """

    def __init__(self, schema: Sequence[Mapping[str, Any]], x_dim: int):
        self.schema = tuple(
            {"name": str(entry["name"]), "levels": tuple(entry["levels"])}
            for entry in schema
        )
        self.x_dim = int(x_dim)
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.coefficients: Optional[Tuple[np.ndarray, ...]] = None
        self.fit_metadata: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        l2: float,
        max_iter: int,
        tolerance_grad: float,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[1] != self.x_dim or X.shape[0] < 1:
            raise ValueError("X has an invalid shape for the softmax baseline.")
        if Y.shape != (X.shape[0], len(self.schema)):
            raise ValueError("Y has an invalid shape for the softmax baseline.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Baseline training inputs must be finite.")
        Y_integer = Y.astype(np.int64)
        if not np.array_equal(Y, Y_integer):
            raise ValueError("Baseline categorical codes must be integers.")
        if not math.isfinite(l2) or l2 <= 0.0 or max_iter < 1:
            raise ValueError("Baseline optimization settings are invalid.")

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std < 1.0e-8] = 1.0
        standardized = (X - self.x_mean) / self.x_std
        design = np.column_stack((np.ones(X.shape[0]), standardized))
        design_tensor = torch.from_numpy(design)
        targets = [
            torch.from_numpy(Y_integer[:, outcome_index]).long()
            for outcome_index in range(len(self.schema))
        ]
        parameters = [
            torch.nn.Parameter(
                torch.zeros(
                    (self.x_dim + 1, len(entry["levels"])), dtype=torch.float64
                )
            )
            for entry in self.schema
        ]
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=int(max_iter),
            tolerance_grad=float(tolerance_grad),
            tolerance_change=1.0e-12,
            history_size=min(50, int(max_iter)),
            line_search_fn="strong_wolfe",
        )

        def objective() -> torch.Tensor:
            optimizer.zero_grad()
            losses = []
            for parameter, target in zip(parameters, targets):
                logits = design_tensor @ parameter
                penalty = 0.5 * l2 * parameter[1:, :].square().sum()
                losses.append(F.cross_entropy(logits, target) + penalty)
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        # LBFGS.step returns the first closure value, not the final objective.
        optimizer.step(objective)
        with torch.no_grad():
            final_terms = []
            for parameter, target in zip(parameters, targets):
                logits = design_tensor @ parameter
                penalty = 0.5 * l2 * parameter[1:, :].square().sum()
                final_terms.append(F.cross_entropy(logits, target) + penalty)
            final_loss = float(torch.stack(final_terms).mean().cpu())
        self.coefficients = tuple(
            parameter.detach().cpu().numpy().copy() for parameter in parameters
        )
        state = optimizer.state.get(parameters[0], {})
        self.fit_metadata = {
            "objective": final_loss,
            "iterations": int(state.get("n_iter", 0)),
            "function_evaluations": int(state.get("func_evals", 0)),
            "l2": float(l2),
            "max_iter": int(max_iter),
            "reached_max_iter": int(state.get("n_iter", max_iter)) >= max_iter,
        }
        return dict(self.fit_metadata)

    def predict_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.coefficients is None or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Fit the independent softmax baseline before prediction.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.x_dim or not np.isfinite(X).all():
            raise ValueError("X has an invalid shape or values.")
        design = np.column_stack((np.ones(X.shape[0]), (X - self.x_mean) / self.x_std))
        return {
            entry["name"]: _softmax(design @ coefficient)
            for entry, coefficient in zip(self.schema, self.coefficients)
        }


def _intercept_probabilities(
    Y_train: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    n_rows: int,
    alpha: float,
) -> Dict[str, np.ndarray]:
    probabilities: Dict[str, np.ndarray] = {}
    for outcome_index, entry in enumerate(schema):
        cardinality = len(entry["levels"])
        counts = np.bincount(
            np.asarray(Y_train[:, outcome_index], dtype=np.int64),
            minlength=cardinality,
        ).astype(np.float64)
        frequency = (counts + alpha) / (Y_train.shape[0] + alpha * cardinality)
        probabilities[entry["name"]] = np.broadcast_to(
            frequency[None, :], (n_rows, cardinality)
        ).copy()
    return probabilities


def fit_probability_cvae(
    scenario: ProbabilityScenario,
    splits: ProbabilitySplits,
    data_seed: int,
    initialization_replicate: int,
    config: ProbabilityStudyConfig,
    device: str,
    verbose: bool,
) -> Tuple[CVAETrainer, Dict[str, Any]]:
    """Fit the one frozen CVAE configuration for a scenario replicate."""
    initialization_seed = _stable_seed(
        PROTOCOL_VERSION,
        scenario.name,
        data_seed,
        "initialization",
        initialization_replicate,
    )
    training_seed = _stable_seed(
        PROTOCOL_VERSION,
        scenario.name,
        data_seed,
        "training",
        initialization_replicate,
    )
    torch.manual_seed(initialization_seed)
    trainer = CVAETrainer(
        x_dim=config.x_dim,
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
        seed=training_seed,
        kl_warmup_epochs=config.kl_warmup_epochs,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_start_epoch=config.early_stopping_start_epoch,
        verbose=verbose,
    )
    return trainer, history


def cvae_marginal_probabilities(
    trainer: CVAETrainer,
    X: np.ndarray,
    *,
    order: int,
    x_batch_size: int,
) -> Dict[str, np.ndarray]:
    """Integrate fitted categorical probabilities over the N(0,I) latent prior."""
    if trainer.outcome_type != "categorical" or not trainer.trained:
        raise ValueError("A fitted categorical CVAETrainer is required.")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != trainer.x_dim:
        raise ValueError("X has the wrong shape for the fitted CVAE.")
    if x_batch_size < 1:
        raise ValueError("x_batch_size must be positive.")
    nodes, weights = tensor_gauss_hermite(order, trainer.latent_dim)
    standardized = trainer._standardize(X)
    z_nodes = torch.from_numpy(nodes.astype(np.float32)).to(trainer.device)
    integration_weights = torch.from_numpy(weights).to(trainer.device, torch.float64)
    result = {
        entry["name"]: np.empty(
            (X.shape[0], len(entry["levels"])), dtype=np.float64
        )
        for entry in trainer.outcome_schema
    }

    trainer.model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], x_batch_size):
            stop = min(start + x_batch_size, X.shape[0])
            batch_rows = stop - start
            x_tensor = torch.from_numpy(standardized[start:stop]).to(trainer.device)
            expanded_x = x_tensor.repeat_interleave(nodes.shape[0], dim=0)
            expanded_z = z_nodes.repeat(batch_rows, 1)
            logits = trainer.model.decode(expanded_x, expanded_z)["logits"].reshape(
                batch_rows, nodes.shape[0], trainer.encoded_y_dim
            )
            for outcome_index, (slice_start, slice_stop) in enumerate(
                trainer.outcome_slices
            ):
                conditional = F.softmax(
                    logits[:, :, slice_start:slice_stop].to(torch.float64), dim=2
                )
                marginal = torch.sum(
                    conditional * integration_weights[None, :, None], dim=1
                )
                name = trainer.outcome_schema[outcome_index]["name"]
                result[name][start:stop] = marginal.cpu().numpy()
    return result


def _probability_validity(
    probabilities: Mapping[str, np.ndarray],
    schema: Sequence[Mapping[str, Any]],
    epsilon: float,
) -> Dict[str, Any]:
    invalid_values = 0
    invalid_row_sums = 0
    floor_hits = 0
    minimum = 1.0
    for entry in schema:
        matrix = np.asarray(probabilities[entry["name"]], dtype=np.float64)
        invalid_values += int(
            np.sum(~np.isfinite(matrix) | (matrix < 0.0) | (matrix > 1.0))
        )
        invalid_row_sums += int(
            np.sum(~np.isclose(matrix.sum(axis=1), 1.0, atol=1.0e-7, rtol=0.0))
        )
        floor_hits += int(np.sum(matrix <= epsilon))
        if matrix.size:
            minimum = min(minimum, float(np.nanmin(matrix)))
    return {
        "invalid_values": invalid_values,
        "invalid_row_sums": invalid_row_sums,
        "floor_hits": floor_hits,
        "minimum_probability": minimum,
        "passed": invalid_values == 0 and invalid_row_sums == 0 and floor_hits == 0,
    }


def _fractional_logistic_calibration(
    actual_probability: np.ndarray,
    fitted_probability: np.ndarray,
    epsilon: float,
) -> Dict[str, Optional[float]]:
    """Fit truth-based intercept/slope without Bernoulli outcome noise."""
    actual = np.asarray(actual_probability, dtype=np.float64)
    fitted = np.clip(np.asarray(fitted_probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    logit = np.log(fitted) - np.log1p(-fitted)
    if float(np.std(logit)) < 1.0e-12:
        return {"intercept": None, "slope": None}
    design = np.column_stack((np.ones(logit.size), logit))
    coefficients = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(60):
        calibrated = _sigmoid(design @ coefficients)
        gradient = design.T @ (calibrated - actual) / actual.size
        variance = np.maximum(calibrated * (1.0 - calibrated), 1.0e-10)
        hessian = (design.T * variance) @ design / actual.size
        hessian += np.eye(2) * 1.0e-10
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
    }


def oracle_expected_probability_metrics(
    probabilities: Mapping[str, np.ndarray],
    oracle_probabilities: Mapping[str, np.ndarray],
    schema: Sequence[Mapping[str, Any]],
    *,
    anchor_outcome: str,
    focal_class_index: int,
    epsilon: float,
) -> Dict[str, Any]:
    """Score fitted probabilities directly against known conditional truth."""
    outcomes: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    focal_record: Optional[Dict[str, Any]] = None
    for entry in schema:
        name = entry["name"]
        fitted = np.asarray(probabilities[name], dtype=np.float64)
        actual = np.asarray(oracle_probabilities[name], dtype=np.float64)
        if fitted.shape != actual.shape:
            raise ValueError(f"Fitted/oracle shapes differ for {name!r}.")
        error = fitted - actual
        clipped = np.clip(fitted, epsilon, 1.0)
        row_kl = np.sum(
            np.where(actual > 0.0, actual * (np.log(actual) - np.log(clipped)), 0.0),
            axis=1,
        )
        row_brier_regret = np.square(error).sum(axis=1)
        row_total_variation = 0.5 * np.abs(error).sum(axis=1)
        outcomes.append(
            {
                "outcome": name,
                "cardinality": int(fitted.shape[1]),
                "mean_kl_regret": float(row_kl.mean()),
                "mean_expected_brier_regret": float(row_brier_regret.mean()),
                "mean_total_variation": float(row_total_variation.mean()),
                "p95_total_variation": float(np.quantile(row_total_variation, 0.95)),
                "probability_mae": float(np.abs(error).mean()),
                "probability_rmse": float(np.sqrt(np.square(error).mean())),
            }
        )
        for class_index, level in enumerate(entry["levels"]):
            class_error = error[:, class_index]
            absolute = np.abs(class_error)
            calibration = _fractional_logistic_calibration(
                actual[:, class_index], fitted[:, class_index], epsilon
            )
            record = {
                "outcome": name,
                "level": str(level),
                "class_index": class_index,
                "mean_true_probability": float(actual[:, class_index].mean()),
                "mean_fitted_probability": float(fitted[:, class_index].mean()),
                "bias": float(class_error.mean()),
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.square(class_error).mean())),
                "absolute_error_p50": float(np.quantile(absolute, 0.50)),
                "absolute_error_p90": float(np.quantile(absolute, 0.90)),
                "absolute_error_p95": float(np.quantile(absolute, 0.95)),
                "absolute_error_p99": float(np.quantile(absolute, 0.99)),
                "truth_calibration_intercept": calibration["intercept"],
                "truth_calibration_slope": calibration["slope"],
            }
            classes.append(record)
            if name == anchor_outcome and class_index == focal_class_index:
                focal_record = dict(record)
    if focal_record is None:
        raise RuntimeError("The focal anchor class was not found in the schema.")
    return {
        "summary": {
            "mean_outcome_kl_regret": float(
                np.mean([item["mean_kl_regret"] for item in outcomes])
            ),
            "mean_outcome_expected_brier_regret": float(
                np.mean([item["mean_expected_brier_regret"] for item in outcomes])
            ),
            "mean_outcome_total_variation": float(
                np.mean([item["mean_total_variation"] for item in outcomes])
            ),
            "maximum_class_mae": float(max(item["mae"] for item in classes)),
        },
        "focal": focal_record,
        "outcomes": outcomes,
        "classes": classes,
    }


def quadrature_convergence_metrics(
    primary: Mapping[str, np.ndarray],
    check: Mapping[str, np.ndarray],
    schema: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    absolute_changes = np.concatenate(
        [
            np.abs(
                np.asarray(check[entry["name"]], dtype=np.float64)
                - np.asarray(primary[entry["name"]], dtype=np.float64)
            ).reshape(-1)
            for entry in schema
        ]
    )
    return {
        "mean_absolute_probability_change": float(absolute_changes.mean()),
        "rmse_probability_change": float(
            np.sqrt(np.square(absolute_changes).mean())
        ),
        "p99_absolute_probability_change": float(
            np.quantile(absolute_changes, 0.99)
        ),
        "maximum_absolute_probability_change": float(absolute_changes.max()),
    }


def engineering_tolerances(focal_prevalence: float) -> Dict[str, float]:
    """Return the predeclared, per-cell probability-recovery tolerances."""
    q = float(focal_prevalence)
    if not 0.0 < q < 1.0:
        raise ValueError("focal_prevalence must lie strictly between zero and one.")
    return {
        "focal_mae_max": max(0.005, 0.10 * q),
        "absolute_focal_bias_max": max(0.0025, 0.05 * q),
        "focal_p95_absolute_error_max": max(0.02, 0.25 * q),
        "mean_outcome_total_variation_max": 0.05,
        "mean_outcome_kl_regret_max": 0.01,
        "quadrature_mean_absolute_change_max": 5.0e-4,
        "quadrature_p99_absolute_change_max": 0.005,
    }


def evaluate_fit_engineering_checks(
    oracle_metrics: Mapping[str, Any],
    quadrature: Mapping[str, float],
    validity: Mapping[str, Any],
    focal_prevalence: float,
) -> Dict[str, Any]:
    """Apply fixed tolerances to one CVAE fit without changing any result."""
    tolerance = engineering_tolerances(focal_prevalence)
    focal = oracle_metrics["focal"]
    summary = oracle_metrics["summary"]
    checks = {
        "focal_mae": {
            "kind": "predictive",
            "value": float(focal["mae"]),
            "threshold": tolerance["focal_mae_max"],
        },
        "absolute_focal_bias": {
            "kind": "predictive",
            "value": abs(float(focal["bias"])),
            "threshold": tolerance["absolute_focal_bias_max"],
        },
        "focal_p95_absolute_error": {
            "kind": "predictive",
            "value": float(focal["absolute_error_p95"]),
            "threshold": tolerance["focal_p95_absolute_error_max"],
        },
        "mean_outcome_total_variation": {
            "kind": "predictive",
            "value": float(summary["mean_outcome_total_variation"]),
            "threshold": tolerance["mean_outcome_total_variation_max"],
        },
        "mean_outcome_kl_regret": {
            "kind": "predictive",
            "value": float(summary["mean_outcome_kl_regret"]),
            "threshold": tolerance["mean_outcome_kl_regret_max"],
        },
        "quadrature_mean_absolute_change": {
            "kind": "prerequisite",
            "value": float(quadrature["mean_absolute_probability_change"]),
            "threshold": tolerance["quadrature_mean_absolute_change_max"],
        },
        "quadrature_p99_absolute_change": {
            "kind": "prerequisite",
            "value": float(quadrature["p99_absolute_probability_change"]),
            "threshold": tolerance["quadrature_p99_absolute_change_max"],
        },
        "no_invalid_or_floor_hit": {
            "kind": "prerequisite",
            "value": int(
                validity["invalid_values"]
                + validity["invalid_row_sums"]
                + validity["floor_hits"]
            ),
            "threshold": 0,
        },
    }
    for check in checks.values():
        check["operator"] = "<="
        check["passed"] = bool(check["value"] <= check["threshold"])
    predictive = [
        check["passed"] for check in checks.values() if check["kind"] == "predictive"
    ]
    prerequisites = [
        check["passed"]
        for check in checks.values()
        if check["kind"] == "prerequisite"
    ]
    return {
        "checks": checks,
        "predictive_passed": bool(all(predictive)),
        "prerequisites_passed": bool(all(prerequisites)),
        "passed": bool(all(predictive) and all(prerequisites)),
        "tolerances": tolerance,
    }


def _compact_cvae_history(history: Mapping[str, Any]) -> Dict[str, Any]:
    best_epoch = history.get("best_epoch")
    best_index = int(best_epoch) - 1 if best_epoch is not None else None

    def value_at(key: str) -> Optional[Any]:
        values = history.get(key)
        if best_index is None or values is None or not 0 <= best_index < len(values):
            return None
        value = values[best_index]
        if isinstance(value, np.ndarray):
            return [float(item) for item in value]
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return float(value)

    return {
        "epochs_ran": int(history.get("epochs_ran", len(history.get("train_loss", [])))),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_validation_loss": value_at("val_loss"),
        "best_train_loss": value_at("train_loss"),
        "best_train_kl_loss": value_at("train_kl_loss"),
        "best_train_kl_per_latent": value_at("train_kl_per_latent"),
        "best_active_latent_units": (
            int(value_at("active_latent_units"))
            if value_at("active_latent_units") is not None
            else None
        ),
    }


def _scenario_factors(scenario: ProbabilityScenario) -> Dict[str, Any]:
    cardinalities = np.asarray(scenario.cardinalities, dtype=np.float64)
    return {
        "design": scenario.design,
        "variant": scenario.variant,
        "n_train": scenario.n_train,
        "n_validation": scenario.n_validation,
        "n_test": scenario.n_test,
        "focal_prevalence": scenario.focal_prevalence,
        "semantic_outcomes": len(scenario.outcomes),
        "minimum_cardinality": int(cardinalities.min()),
        "maximum_cardinality": int(cardinalities.max()),
        "summed_cardinality": int(cardinalities.sum()),
        "cardinality_heterogeneity": float(cardinalities.std(ddof=0)),
        "homogeneous_cardinality": (
            int(cardinalities[0]) if np.all(cardinalities == cardinalities[0]) else None
        ),
        "copula_rho": scenario.copula_rho,
    }


def _is_initialization_sensitivity_scenario(
    scenario: ProbabilityScenario,
    config: ProbabilityStudyConfig,
) -> bool:
    if scenario.design != "core":
        return False
    combinations = {
        (
            min(config.sample_sizes),
            min(config.focal_prevalences),
            max(config.homogeneous_cardinalities),
        ),
        (
            config.rho_control_n_train,
            config.rho_control_prevalence,
            config.rho_control_cardinality,
        ),
        (
            max(config.sample_sizes),
            max(config.focal_prevalences),
            min(config.homogeneous_cardinalities),
        ),
    }
    return (
        scenario.n_train,
        scenario.focal_prevalence,
        scenario.cardinalities[0],
    ) in combinations


def _is_plot_sentinel(
    scenario: ProbabilityScenario,
    data_seed: int,
    initialization_replicate: int,
    config: ProbabilityStudyConfig,
) -> bool:
    if data_seed != config.data_seeds[0] or initialization_replicate != 0:
        return False
    if _is_initialization_sensitivity_scenario(scenario, config):
        return True
    return (
        scenario.design == "heterogeneity"
        and scenario.n_train == config.rho_control_n_train
    ) or scenario.design == "rho_control"


def _copy_probability_rows(
    probabilities: Mapping[str, np.ndarray], rows: int
) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray(matrix[:rows], dtype=np.float64).copy()
        for name, matrix in probabilities.items()
    }


def compact_probability_diagnostics(
    comparison: Mapping[str, Any],
    *,
    anchor_outcome: str,
    focal_class_index: int,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Strip repeated plotting arrays while retaining all scalar diagnostics.

    Reliability-bin records for every class/model/cell would make the canonical
    JSON unnecessarily large.  Predeclared sentinel payloads retain raw rows
    for plots.  True-probability-band results are retained only for the focal
    anchor class, which is the prespecified rarity estimand.
    """
    compact_models: Dict[str, Any] = {}
    focal_bands: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, diagnostics in comparison["models"].items():
        compact_models[model_name] = {
            "summary": dict(diagnostics["summary"]),
            "outcomes": [dict(item) for item in diagnostics["outcomes"]],
            "classes": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "reliability_bins"
                }
                for item in diagnostics["classes"]
            ],
            "calibration": dict(diagnostics["calibration"]),
        }
        focal_bands[model_name] = [
            dict(item)
            for item in diagnostics["true_probability_bands"]
            if item["outcome"] == anchor_outcome
            and int(item["class_index"]) == focal_class_index
        ]
    return (
        {
            "models": compact_models,
            "schema": [dict(entry) for entry in comparison["schema"]],
            "calibration": dict(comparison["calibration"]),
            "has_oracle": bool(comparison["has_oracle"]),
            "compaction": (
                "scalar diagnostics retained; repeated reliability bins removed; "
                "focal true-probability bands stored separately; sentinel raw rows "
                "support calibration plots"
            ),
        },
        focal_bands,
    )


def evaluate_probability_scenario_seed(
    scenario: ProbabilityScenario,
    data_seed: int,
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
    *,
    initialization_replicate: int = 0,
    device: str = "cpu",
    retain_plot_payload: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit and score one declared scenario/data/initialization replicate."""
    started = time.perf_counter()
    splits = simulate_probability_splits(scenario, data_seed, config)
    trainer, cvae_history = fit_probability_cvae(
        scenario,
        splits,
        data_seed,
        initialization_replicate,
        config,
        device,
        verbose,
    )
    cvae_probabilities = cvae_marginal_probabilities(
        trainer,
        splits.test.X,
        order=config.fitted_quadrature_order,
        x_batch_size=config.quadrature_x_batch_size,
    )
    check_rows = min(config.quadrature_check_rows, scenario.n_test)
    check_probabilities = cvae_marginal_probabilities(
        trainer,
        splits.test.X[:check_rows],
        order=config.fitted_quadrature_check_order,
        x_batch_size=config.quadrature_x_batch_size,
    )
    primary_subset = {
        name: values[:check_rows] for name, values in cvae_probabilities.items()
    }
    quadrature = quadrature_convergence_metrics(
        primary_subset, check_probabilities, scenario.outcome_schema
    )

    baseline = IndependentSoftmaxBaseline(scenario.outcome_schema, config.x_dim)
    baseline_fit = baseline.fit(
        splits.train.X,
        splits.train.Y,
        l2=config.baseline_l2,
        max_iter=config.baseline_max_iter,
        tolerance_grad=config.baseline_tolerance_grad,
    )
    baseline_probabilities = baseline.predict_probabilities(splits.test.X)
    intercept_probabilities = _intercept_probabilities(
        splits.train.Y,
        scenario.outcome_schema,
        scenario.n_test,
        config.intercept_alpha,
    )
    model_probabilities: Dict[str, Mapping[str, np.ndarray]] = {
        "cvae": cvae_probabilities,
        "independent_softmax": baseline_probabilities,
        "intercept_null": intercept_probabilities,
        "oracle": splits.test.oracle_probabilities,
    }
    comparison = compare_probability_models(
        model_probabilities,
        splits.test.Y,
        scenario.outcome_schema,
        oracle_probabilities=splits.test.oracle_probabilities,
        n_bins=config.calibration_bins,
        calibration_strategy="quantile",
        epsilon=config.probability_epsilon,
    )
    compact_diagnostics, focal_probability_bands = compact_probability_diagnostics(
        comparison,
        anchor_outcome=scenario.anchor.name,
        focal_class_index=int(scenario.anchor.focal_class_index),
    )
    oracle_metrics = {
        model_name: oracle_expected_probability_metrics(
            probabilities,
            splits.test.oracle_probabilities,
            scenario.outcome_schema,
            anchor_outcome=scenario.anchor.name,
            focal_class_index=int(scenario.anchor.focal_class_index),
            epsilon=config.probability_epsilon,
        )
        for model_name, probabilities in model_probabilities.items()
    }
    validity = _probability_validity(
        cvae_probabilities, scenario.outcome_schema, config.probability_epsilon
    )
    engineering_checks = evaluate_fit_engineering_checks(
        oracle_metrics["cvae"],
        quadrature,
        validity,
        scenario.focal_prevalence,
    )
    training_counts = {
        entry["name"]: np.bincount(
            splits.train.Y[:, outcome_index], minlength=len(entry["levels"])
        ).astype(int).tolist()
        for outcome_index, entry in enumerate(scenario.outcome_schema)
    }
    focal_diagnostic = next(
        item
        for item in comparison["models"]["cvae"]["classes"]
        if item["outcome"] == scenario.anchor.name
        and item["class_index"] == scenario.anchor.focal_class_index
    )

    record: Dict[str, Any] = {
        "scenario": scenario.name,
        "data_seed": int(data_seed),
        "initialization_replicate": int(initialization_replicate),
        "is_initialization_sensitivity": initialization_replicate > 0,
        "factors": _scenario_factors(scenario),
        "cardinalities": list(scenario.cardinalities),
        "schema": scenario.outcome_schema,
        "anchor_outcome": scenario.anchor.name,
        "focal_class_index": int(scenario.anchor.focal_class_index),
        "split_integrity": True,
        "training_counts": training_counts,
        "focal_training_count": int(
            training_counts[scenario.anchor.name][scenario.anchor.focal_class_index]
        ),
        "focal_test_count": int(focal_diagnostic["event_count"]),
        "cvae_training": _compact_cvae_history(cvae_history),
        "independent_softmax_training": baseline_fit,
        "model_diagnostics": compact_diagnostics,
        "focal_true_probability_bands": focal_probability_bands,
        "oracle_metrics": oracle_metrics,
        "quadrature_convergence": quadrature,
        "probability_validity": validity,
        "engineering_checks": engineering_checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if retain_plot_payload:
        plot_rows = min(config.plot_rows, scenario.n_test)
        anchor_reliability_bins = {}
        for model_name in ("cvae", "independent_softmax"):
            anchor_reliability_bins[model_name] = [
                {
                    "outcome": item["outcome"],
                    "level": item["level"],
                    "class_index": int(item["class_index"]),
                    "n": int(item["n"]),
                    "event_count": int(item["event_count"]),
                    "reliability_bins": [
                        dict(bin_record) for bin_record in item["reliability_bins"]
                    ],
                }
                for item in comparison["models"][model_name]["classes"]
                if item["outcome"] == scenario.anchor.name
            ]
        record["plot_payload"] = {
            "selection": (
                "first fixed test rows; sentinel scenarios and the first predeclared "
                "data seed were selected before fitting"
            ),
            "rows": plot_rows,
            "Y": splits.test.Y[:plot_rows].copy(),
            "anchor_reliability_bins_full_test": anchor_reliability_bins,
            "oracle_probabilities": _copy_probability_rows(
                splits.test.oracle_probabilities, plot_rows
            ),
            "model_probabilities": {
                name: _copy_probability_rows(probabilities, plot_rows)
                for name, probabilities in model_probabilities.items()
                if name != "oracle"
            },
        }
    return record


def aggregate_cell_engineering_status(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize the five base replicates for each cell; no global verdict."""
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if int(record["initialization_replicate"]) != 0:
            continue
        grouped.setdefault(str(record["scenario"]), []).append(record)
    statuses: List[Dict[str, Any]] = []
    for scenario_name, cell in grouped.items():
        prevalence = float(cell[0]["factors"]["focal_prevalence"])
        tolerance = engineering_tolerances(prevalence)

        def mean(path: Tuple[str, ...]) -> float:
            values = []
            for record in cell:
                value: Any = record
                for key in path:
                    value = value[key]
                values.append(float(value))
            return float(np.mean(values))

        focal_biases = [
            abs(float(record["oracle_metrics"]["cvae"]["focal"]["bias"]))
            for record in cell
        ]
        values = {
            "focal_mae": mean(("oracle_metrics", "cvae", "focal", "mae")),
            "absolute_focal_bias": float(np.mean(focal_biases)),
            "focal_p95_absolute_error": mean(
                ("oracle_metrics", "cvae", "focal", "absolute_error_p95")
            ),
            "mean_outcome_total_variation": mean(
                ("oracle_metrics", "cvae", "summary", "mean_outcome_total_variation")
            ),
            "mean_outcome_kl_regret": mean(
                ("oracle_metrics", "cvae", "summary", "mean_outcome_kl_regret")
            ),
            # Numerical prerequisites must hold in every replicate, so the
            # cell statistic is the worst (largest) seed-level discrepancy.
            "quadrature_mean_absolute_change": max(
                float(record["quadrature_convergence"]["mean_absolute_probability_change"])
                for record in cell
            ),
            "quadrature_p99_absolute_change": max(
                float(record["quadrature_convergence"]["p99_absolute_probability_change"])
                for record in cell
            ),
            "invalid_or_floor_hits": sum(
                int(record["probability_validity"]["invalid_values"])
                + int(record["probability_validity"]["invalid_row_sums"])
                + int(record["probability_validity"]["floor_hits"])
                for record in cell
            ),
        }
        checks = {
            "focal_mae": values["focal_mae"] <= tolerance["focal_mae_max"],
            "absolute_focal_bias": values["absolute_focal_bias"]
            <= tolerance["absolute_focal_bias_max"],
            "focal_p95_absolute_error": values["focal_p95_absolute_error"]
            <= tolerance["focal_p95_absolute_error_max"],
            "mean_outcome_total_variation": values["mean_outcome_total_variation"]
            <= tolerance["mean_outcome_total_variation_max"],
            "mean_outcome_kl_regret": values["mean_outcome_kl_regret"]
            <= tolerance["mean_outcome_kl_regret_max"],
            "quadrature_mean_absolute_change": values[
                "quadrature_mean_absolute_change"
            ]
            <= tolerance["quadrature_mean_absolute_change_max"],
            "quadrature_p99_absolute_change": values[
                "quadrature_p99_absolute_change"
            ]
            <= tolerance["quadrature_p99_absolute_change_max"],
            "no_invalid_or_floor_hit": values["invalid_or_floor_hits"] == 0,
        }
        replicate_pass_count = sum(
            bool(record["engineering_checks"]["passed"]) for record in cell
        )
        worst_replicate_values = {
            "focal_mae": max(
                float(record["oracle_metrics"]["cvae"]["focal"]["mae"])
                for record in cell
            ),
            "absolute_focal_bias": max(focal_biases),
            "focal_p95_absolute_error": max(
                float(
                    record["oracle_metrics"]["cvae"]["focal"][
                        "absolute_error_p95"
                    ]
                )
                for record in cell
            ),
            "mean_outcome_total_variation": max(
                float(
                    record["oracle_metrics"]["cvae"]["summary"][
                        "mean_outcome_total_variation"
                    ]
                )
                for record in cell
            ),
            "mean_outcome_kl_regret": max(
                float(
                    record["oracle_metrics"]["cvae"]["summary"][
                        "mean_outcome_kl_regret"
                    ]
                )
                for record in cell
            ),
        }
        all_replicates_passed = replicate_pass_count == len(cell)
        statuses.append(
            {
                "scenario": scenario_name,
                "n_replicates": len(cell),
                "factors": dict(cell[0]["factors"]),
                "values": values,
                "tolerances": tolerance,
                "checks": checks,
                "replicate_pass_count": int(replicate_pass_count),
                "all_replicates_passed": all_replicates_passed,
                "worst_replicate_values": worst_replicate_values,
                "passed": bool(all(checks.values()) and all_replicates_passed),
            }
        )
    return statuses


def build_probability_protocol_manifest(
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Return the exact, hashable protocol including all generated coefficients."""
    scenarios = build_probability_scenarios(config)
    return {
        "protocol": PROTOCOL_VERSION,
        "question": (
            "How accurately are individual marginal conditional probabilities "
            "recovered across sample size, focal prevalence, cardinality, and "
            "matched cardinality heterogeneity?"
        ),
        "config": asdict(config),
        "dgp_contract": {
            "X": "iid Uniform[-1,1] in three dimensions",
            "marginals": "exact linear softmax",
            "dependence": (
                "equicorrelated Gaussian copula; V_j=sqrt(rho)S+"
                "sqrt(1-rho)E_j and U_j=Phi(V_j)"
            ),
            "focal_prevalence": (
                "population E_X[P(anchor=focal|X)] solved by deterministic "
                "tensor Gauss-Legendre quadrature"
            ),
            "context_prevalence": (
                "exactly balanced in the population by deterministic multinomial "
                "intercept calibration"
            ),
        },
        "primary_estimands": {
            "focal": "bias, MAE, RMSE, and p95 absolute p_hat-p_true error",
            "all_outcomes": (
                "equal-outcome mean total variation, expected Brier regret, "
                "and KL/log-score regret"
            ),
            "realized_secondary": (
                "NLL, multiclass Brier, one-vs-rest ROC AUC, average precision, "
                "and adaptive-bin empirical reliability"
            ),
            "cvae_probability": (
                "E_Z[softmax(decoder(X,Z))] under the fitted standard-normal prior"
            ),
        },
        "engineering_tolerances": {
            "focal_mae": "<= max(0.005, 0.10*q)",
            "absolute_focal_bias": "<= max(0.0025, 0.05*q)",
            "focal_p95_absolute_error": "<= max(0.02, 0.25*q)",
            "mean_outcome_total_variation": "<= 0.05",
            "mean_outcome_kl_regret": "<= 0.01 nat/outcome",
            "quadrature_mean_absolute_change": "GH21 vs GH31 <= 0.0005",
            "quadrature_p99_absolute_change": "GH21 vs GH31 <= 0.005",
            "probability_validity": "no invalid values, invalid row sums, or floor hits",
            "scope": (
                "per-cell engineering envelope only; ROC AUC and average precision "
                "are descriptive and there is no global pass/fail claim"
            ),
        },
        "replication": {
            "base": "five fresh fixed data/training seeds in every cell",
            "initialization_sensitivity": (
                "three total initialization/training replicates on the first fixed "
                "data split for predeclared worst, central, and favorable core cells"
            ),
            "plot_payload": (
                "first plot_rows test rows for predeclared sentinel cells at the "
                "first fixed data seed and initialization replicate zero"
            ),
        },
        "development_history": {
            "retired_proposed_seed_set": [4243, 8677, 16127, 32353, 65537],
            "seen_development_checks": [
                (
                    "An implementation-only CLI smoke fit was run for the rho=0 "
                    "control with proposed seed 4243. It completed successfully as "
                    "software but did not meet the complete engineering envelope "
                    "(0 of 1 evaluated cells passed)."
                ),
                (
                    "A CVAE smoke fit for retired seed 4243 on "
                    "core_n500_q010_k2 produced focal MAE about 0.0092, bias about "
                    "+0.0049, and p95 absolute error about 0.0227."
                ),
                (
                    "Focal training counts were inspected across the entire retired "
                    "proposed seed set, and baseline-only software checks were run "
                    "on several retired cells."
                ),
            ],
            "disposition": (
                "The entire proposed seed set was retired. No DGP coefficient, "
                "model setting, diagnostic, or engineering tolerance was changed "
                "in response. The replacement canonical seeds were reserved unseen "
                "until after source commit and public protocol posting."
            ),
            "canonical_seed_set": list(DEFAULT_DATA_SEEDS),
        },
        "reproducibility_metadata_fields": (
            "Python, NumPy, Torch, platform, git HEAD/dirty state, imported package "
            "path, CVAETrainer source path/hash, and runner source path/hash are "
            "captured at execution time outside this environment-independent hash"
        ),
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }


def probability_protocol_sha256(
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
) -> str:
    manifest = build_probability_protocol_manifest(config)
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_probability_environment_metadata() -> Dict[str, Any]:
    """Record enough source/runtime identity to audit a generated report."""
    repository_root = Path(__file__).resolve().parents[1]
    runner_path = Path(__file__).resolve()
    trainer_source_raw = inspect.getsourcefile(CVAETrainer)
    trainer_path = Path(trainer_source_raw).resolve() if trainer_source_raw else None
    package_path = (
        Path(multioutcome_cvae_package.__file__).resolve()
        if multioutcome_cvae_package.__file__ is not None
        else None
    )

    def relative(path: Optional[Path]) -> str:
        if path is None:
            return "unknown"
        try:
            return str(path.relative_to(repository_root))
        except ValueError:
            return str(path)

    def sha256(path: Optional[Path]) -> str:
        if path is None:
            return "unknown"
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return "unknown"

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git",) + arguments,
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    dirty = git("status", "--porcelain")
    try:
        package_version = importlib_metadata.version("multioutcome-cvae")
    except importlib_metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "package_version": package_version,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": "unknown" if dirty == "unknown" else bool(dirty),
        "package_path": relative(package_path),
        "trainer_source_path": relative(trainer_path),
        "trainer_source_sha256": sha256(trainer_path),
        "runner_source_path": relative(runner_path),
        "runner_source_sha256": sha256(runner_path),
    }


def run_probability_validation(
    config: ProbabilityStudyConfig = DEFAULT_CONFIG,
    *,
    device: str = "cpu",
    scenario_names: Optional[Sequence[str]] = None,
    data_seeds: Optional[Sequence[int]] = None,
    include_plot_payloads: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run selected cells, retaining every result whether favorable or not."""
    config.validate()
    all_scenarios = build_probability_scenarios(config)
    scenario_by_name = {scenario.name: scenario for scenario in all_scenarios}
    if scenario_names is None:
        scenarios = all_scenarios
    else:
        missing = sorted(set(scenario_names).difference(scenario_by_name))
        if missing:
            raise ValueError(f"Unknown scenario names: {missing}.")
        scenarios = tuple(scenario_by_name[name] for name in scenario_names)
    selected_seeds = (
        tuple(int(seed) for seed in data_seeds)
        if data_seeds is not None
        else config.data_seeds
    )
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Selected data seeds must be nonempty and unique.")

    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for data_seed in selected_seeds:
            if verbose:
                print(f"[{scenario.name}] data_seed={data_seed} init=0", flush=True)
            records.append(
                evaluate_probability_scenario_seed(
                    scenario,
                    data_seed,
                    config,
                    initialization_replicate=0,
                    device=device,
                    retain_plot_payload=(
                        include_plot_payloads
                        and _is_plot_sentinel(scenario, data_seed, 0, config)
                    ),
                    verbose=False,
                )
            )

    # These are additional fits of the same fixed data, not additional data
    # replicates.  They are deliberately separate from the five-seed cell means.
    if (
        config.include_initialization_sensitivity
        and config.initialization_sensitivity_replicates > 1
        and config.data_seeds[0] in selected_seeds
    ):
        for scenario in scenarios:
            if not _is_initialization_sensitivity_scenario(scenario, config):
                continue
            for initialization_replicate in range(
                1, config.initialization_sensitivity_replicates
            ):
                if verbose:
                    print(
                        f"[{scenario.name}] data_seed={config.data_seeds[0]} "
                        f"init={initialization_replicate}",
                        flush=True,
                    )
                records.append(
                    evaluate_probability_scenario_seed(
                        scenario,
                        config.data_seeds[0],
                        config,
                        initialization_replicate=initialization_replicate,
                        device=device,
                        retain_plot_payload=False,
                        verbose=False,
                    )
                )

    manifest = build_probability_protocol_manifest(config)
    return {
        "protocol": PROTOCOL_VERSION,
        "protocol_manifest_sha256": probability_protocol_sha256(config),
        "protocol_manifest": manifest,
        "development_history": dict(manifest["development_history"]),
        "config": asdict(config),
        "environment": collect_probability_environment_metadata(),
        "device": device,
        "selected_scenarios": [scenario.name for scenario in scenarios],
        "selected_data_seeds": list(selected_seeds),
        "is_complete_canonical_grid": bool(
            config == DEFAULT_CONFIG
            and tuple(scenario.name for scenario in scenarios)
            == tuple(scenario.name for scenario in all_scenarios)
            and selected_seeds == config.data_seeds
        ),
        "records": records,
        "cell_engineering_status": aggregate_cell_engineering_status(records),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_probability_results(run: Mapping[str, Any], output_path: Path) -> Path:
    """Write a portable JSON result used by the report/plot renderer."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output. The report layer may instead consume the run in memory.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device; CPU is the canonical default."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Run only this exact scenario name (repeatable; otherwise run the full grid).",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Run only this data seed (repeatable; otherwise use all five fixed seeds).",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true", help="List fixed scenario names and exit."
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.list_scenarios:
        for scenario in build_probability_scenarios(DEFAULT_CONFIG):
            print(scenario.name)
        return 0
    run = run_probability_validation(
        DEFAULT_CONFIG,
        device=arguments.device,
        scenario_names=arguments.scenarios,
        data_seeds=arguments.seeds,
        include_plot_payloads=True,
        verbose=arguments.verbose,
    )
    if arguments.output is not None:
        written = write_probability_results(run, arguments.output)
        print(f"Wrote {written}")
    passed = sum(item["passed"] for item in run["cell_engineering_status"])
    total = len(run["cell_engineering_status"])
    print(
        f"Completed {len(run['records'])} fits in {run['runtime_seconds']:.1f}s; "
        f"{passed}/{total} evaluated cells met their engineering tolerances."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
