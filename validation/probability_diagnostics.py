"""Probability-focused diagnostics and plots for categorical simulations.

This module deliberately depends only on NumPy for metric computation.  Plotting
is lazy and requires the project's optional ``plot`` extra (Matplotlib).  The
functions accept the same named probability-matrix contract as
``CVAETrainer.predict_params()`` and make no assumptions about how a model was
fitted.

Two different comparisons are kept separate:

* reliability compares predicted probabilities with realized zero/one events;
* oracle agreement, available in simulation, compares every fitted probability
  directly with the data-generating conditional probability for the same row.

The distinction matters: an observed outcome is not the "actual probability"
for one person.  Reliability estimates calibration in groups, whereas the
oracle comparison measures row-level probability error directly.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TRUE_PROBABILITY_BANDS: Tuple[float, ...] = (
    0.0,
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
    1.0,
)
_WILSON_Z_95 = 1.959963984540054


def _normalise_schema(
    schema: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(schema, Sequence) or isinstance(schema, (str, bytes)):
        raise TypeError("schema must be a sequence of outcome mappings.")
    normalised: List[Dict[str, Any]] = []
    names = set()
    for outcome_index, entry in enumerate(schema):
        if not isinstance(entry, Mapping):
            raise TypeError(f"schema[{outcome_index}] must be a mapping.")
        name = entry.get("name")
        levels = entry.get("levels")
        if not isinstance(name, str) or not name:
            raise ValueError(f"schema[{outcome_index}] requires a nonempty name.")
        if name in names:
            raise ValueError(f"Duplicate outcome name {name!r}.")
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise TypeError(f"Outcome {name!r} levels must be a sequence.")
        if len(levels) < 2:
            raise ValueError(f"Outcome {name!r} must have at least two levels.")
        level_strings = tuple(str(level) for level in levels)
        if len(set(level_strings)) != len(level_strings):
            raise ValueError(
                f"Outcome {name!r} has level labels that are not unique as strings."
            )
        names.add(name)
        normalised.append({"name": name, "levels": level_strings})
    if not normalised:
        raise ValueError("schema must contain at least one outcome.")
    return tuple(normalised)


def _validate_codes(Y: np.ndarray, schema: Sequence[Mapping[str, Any]]) -> np.ndarray:
    array = np.asarray(Y)
    if array.ndim != 2 or array.shape[1] != len(schema):
        raise ValueError(
            f"Y must have shape (n, {len(schema)}); received {array.shape}."
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError("Y must contain at least one row of finite codes.")
    integer = array.astype(np.int64)
    if not np.array_equal(array, integer):
        raise ValueError("Y must contain integer category codes.")
    for outcome_index, entry in enumerate(schema):
        cardinality = len(entry["levels"])
        column = integer[:, outcome_index]
        if np.any(column < 0) or np.any(column >= cardinality):
            raise ValueError(
                f"Y codes for {entry['name']!r} must be in [0, {cardinality})."
            )
    return integer


def _validate_probability_mapping(
    probabilities: Mapping[str, np.ndarray],
    n_rows: int,
    schema: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, np.ndarray]:
    if not isinstance(probabilities, Mapping):
        raise TypeError(f"{label} must be a mapping of names to matrices.")
    expected_names = {entry["name"] for entry in schema}
    if set(probabilities) != expected_names:
        missing = sorted(expected_names.difference(probabilities))
        extra = sorted(set(probabilities).difference(expected_names))
        raise ValueError(
            f"{label} names differ from schema; missing={missing}, extra={extra}."
        )
    validated: Dict[str, np.ndarray] = {}
    for entry in schema:
        name = entry["name"]
        matrix = np.asarray(probabilities[name], dtype=np.float64)
        expected_shape = (n_rows, len(entry["levels"]))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{label}[{name!r}] must have shape {expected_shape}; "
                f"received {matrix.shape}."
            )
        if (
            not np.isfinite(matrix).all()
            or np.any(matrix < 0.0)
            or np.any(matrix > 1.0)
        ):
            raise ValueError(f"{label}[{name!r}] contains invalid probabilities.")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
            raise ValueError(f"Rows of {label}[{name!r}] must sum to one.")
        validated[name] = matrix
    return validated


def _validate_true_probability_bands(
    bands: Sequence[float],
) -> Tuple[float, ...]:
    values = tuple(float(value) for value in bands)
    if len(values) < 2 or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("true_probability_bands must start at 0 and end at 1.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("true_probability_bands must be finite.")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("true_probability_bands must be strictly increasing.")
    return values


def _binary_ranking_metrics(
    observed: np.ndarray, probability: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """Return ROC AUC and average precision from one shared stable sort.

    Tied scores are reduced as complete threshold groups.  Keeping the group
    arithmetic in NumPy avoids a Python loop per distinct fitted probability,
    which is material for the large high-cardinality validation grid.
    """
    y = np.asarray(observed)
    score = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or score.ndim != 1 or y.shape != score.shape:
        raise ValueError("observed and probability must be same-length vectors.")
    if y.size < 1 or not np.isfinite(score).all():
        raise ValueError("Ranking-metric inputs must be nonempty and finite.")
    if not np.all(np.logical_or(y == 0, y == 1)):
        raise ValueError("observed must contain only zeroes and ones.")
    y = y.astype(np.int8)
    positives = int(y.sum())
    negatives = int(y.size - positives)

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order].astype(np.int64, copy=False)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.array([y.size], dtype=np.int64)))
    group_sizes = stops - starts
    group_positives = np.add.reduceat(sorted_y, starts)

    auc: Optional[float]
    if positives == 0 or negatives == 0:
        auc = None
    else:
        # One-based average ranks for each complete tied-score group.
        average_ranks = ((starts + 1) + stops) / 2.0
        positive_rank_sum = float(np.sum(group_positives * average_ranks))
        auc = float(
            (positive_rank_sum - positives * (positives + 1) / 2.0)
            / (positives * negatives)
        )

    average_precision: Optional[float]
    if positives == 0:
        average_precision = None
    else:
        # Descending score thresholds are obtained by reversing the ascending
        # tie groups; no second argsort is needed.
        descending_positives = group_positives[::-1]
        descending_sizes = group_sizes[::-1]
        cumulative_positives = np.cumsum(descending_positives)
        cumulative_selected = np.cumsum(descending_sizes)
        average_precision = float(
            np.sum(
                (descending_positives / positives)
                * (cumulative_positives / cumulative_selected)
            )
        )
    return auc, average_precision


def binary_auc(observed: np.ndarray, probability: np.ndarray) -> Optional[float]:
    """Return exact empirical ROC AUC using average ranks for tied scores.

    ``None`` is returned when the sample has no positive or no negative event;
    reporting a numeric AUC in that case would be misleading.
    """

    auc, _ = _binary_ranking_metrics(observed, probability)
    return auc


def binary_pr_auc(observed: np.ndarray, probability: np.ndarray) -> Optional[float]:
    """Return non-interpolated average precision (step-area PR-AUC).

    Thresholds are evaluated only after complete tied-score groups, so the
    result is invariant to the ordering of observations with equal scores.
    ``None`` is returned when no positive event is observed.
    """

    _, average_precision = _binary_ranking_metrics(observed, probability)
    return average_precision


def _wilson_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials < 1:
        raise ValueError("Wilson interval requires at least one trial.")
    proportion = successes / trials
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def calibration_curve(
    observed: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    oracle_probability: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Estimate an observed-versus-predicted reliability curve.

    Quantile bins are the default because they keep useful effective sample
    sizes for rare categories.  ``strategy='uniform'`` instead uses fixed-width
    intervals on [0, 1].  Empty intervals are omitted in both cases.
    """

    y = np.asarray(observed)
    probability = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 1 or y.shape != probability.shape:
        raise ValueError("observed and probability must be same-length vectors.")
    if y.size < 1 or not np.isfinite(probability).all():
        raise ValueError("Calibration inputs must be nonempty and finite.")
    if not np.all(np.logical_or(y == 0, y == 1)):
        raise ValueError("observed must contain only zeroes and ones.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("probability values must lie in [0, 1].")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")
    if strategy not in ("quantile", "uniform"):
        raise ValueError("strategy must be 'quantile' or 'uniform'.")
    oracle = None
    if oracle_probability is not None:
        oracle = np.asarray(oracle_probability, dtype=np.float64)
        if oracle.shape != probability.shape or not np.isfinite(oracle).all():
            raise ValueError(
                "oracle_probability must be finite and have the same shape as probability."
            )
        if np.any(oracle < 0.0) or np.any(oracle > 1.0):
            raise ValueError("oracle_probability values must lie in [0, 1].")

    if strategy == "uniform":
        bin_index = np.minimum((probability * n_bins).astype(np.int64), n_bins - 1)
        groups = [np.flatnonzero(bin_index == index) for index in range(n_bins)]
    else:
        quantiles = np.linspace(0.0, 1.0, min(n_bins, y.size) + 1)
        try:
            edges = np.quantile(probability, quantiles, method="linear")
        except TypeError:  # NumPy 1.20/1.21 compatibility.
            edges = np.quantile(probability, quantiles, interpolation="linear")
        edges = np.unique(edges)
        if edges.size == 1:
            groups = [np.arange(y.size)]
        else:
            bin_index = np.searchsorted(edges[1:-1], probability, side="right")
            groups = [
                np.flatnonzero(bin_index == index) for index in range(edges.size - 1)
            ]

    bins: List[Dict[str, Any]] = []
    for group in groups:
        if group.size == 0:
            continue
        event_count = int(y[group].sum())
        lower, upper = _wilson_interval(event_count, int(group.size))
        predicted_mean = float(probability[group].mean())
        observed_fraction = float(event_count / group.size)
        item = {
            "count": int(group.size),
            "event_count": event_count,
            "minimum_predicted": float(probability[group].min()),
            "maximum_predicted": float(probability[group].max()),
            "mean_predicted": predicted_mean,
            "observed_fraction": observed_fraction,
            "observed_wilson_lower_95": lower,
            "observed_wilson_upper_95": upper,
            "signed_gap": observed_fraction - predicted_mean,
        }
        if oracle is not None:
            item["mean_oracle_probability"] = float(oracle[group].mean())
            item["oracle_minus_predicted"] = float(
                oracle[group].mean() - predicted_mean
            )
        bins.append(item)
    return bins


def _safe_binary_log_loss(
    observed: np.ndarray, probability: np.ndarray, epsilon: float
) -> float:
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(observed * np.log(clipped) + (1.0 - observed) * np.log1p(-clipped))
    )


def _oracle_kl_rows(
    oracle_probability: np.ndarray,
    fitted_probability: np.ndarray,
) -> np.ndarray:
    """Return exact rowwise KL(oracle || fitted), including infinite regret.

    Zero-probability oracle cells contribute zero.  A fitted zero where the
    oracle assigns positive mass has infinite log-score regret; silently
    flooring that case would no longer be the advertised KL divergence.
    """

    oracle = np.asarray(oracle_probability, dtype=np.float64)
    fitted = np.asarray(fitted_probability, dtype=np.float64)
    positive_oracle = oracle > 0.0
    contributions = np.zeros_like(oracle)
    with np.errstate(divide="ignore", invalid="ignore"):
        contributions[positive_oracle] = oracle[positive_oracle] * (
            np.log(oracle[positive_oracle]) - np.log(fitted[positive_oracle])
        )
    row_kl = contributions.sum(axis=1)
    # Properly normalized inputs make KL nonnegative.  Remove only negative
    # roundoff, while retaining positive infinity for unsupported truth.
    finite = np.isfinite(row_kl)
    row_kl[finite] = np.maximum(row_kl[finite], 0.0)
    return row_kl


def probability_diagnostics(
    probabilities: Mapping[str, np.ndarray],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    *,
    oracle_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
    true_probability_bands: Sequence[float] = DEFAULT_TRUE_PROBABILITY_BANDS,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Compute JSON-friendly marginal probability diagnostics.

    AUC, calibration, and binary Brier scores are reported one class versus the
    rest.  Multiclass Brier and negative log likelihood are reported once per
    semantic outcome.  When DGP oracle probabilities are supplied, MAE/RMSE are
    computed for individual row/class probabilities, including prespecified
    true-probability bands.
    """

    normalised_schema = _normalise_schema(schema)
    codes = _validate_codes(Y, normalised_schema)
    fitted = _validate_probability_mapping(
        probabilities, codes.shape[0], normalised_schema, "probabilities"
    )
    oracle = None
    if oracle_probabilities is not None:
        oracle = _validate_probability_mapping(
            oracle_probabilities,
            codes.shape[0],
            normalised_schema,
            "oracle_probabilities",
        )
    bands = _validate_true_probability_bands(true_probability_bands)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie strictly between 0 and 0.5.")

    outcome_records: List[Dict[str, Any]] = []
    class_records: List[Dict[str, Any]] = []
    band_records: List[Dict[str, Any]] = []
    rows = np.arange(codes.shape[0])

    for outcome_index, entry in enumerate(normalised_schema):
        name = entry["name"]
        levels = entry["levels"]
        matrix = fitted[name]
        observed_matrix = np.eye(len(levels), dtype=np.float64)[codes[:, outcome_index]]
        selected = matrix[rows, codes[:, outcome_index]]
        multiclass_brier_rows = np.square(matrix - observed_matrix).sum(axis=1)
        outcome_record: Dict[str, Any] = {
            "outcome": name,
            "cardinality": len(levels),
            "n": int(codes.shape[0]),
            "nll": float(-np.log(np.clip(selected, epsilon, 1.0)).mean()),
            "multiclass_brier": float(multiclass_brier_rows.mean()),
            "probability_floor_epsilon": float(epsilon),
            "fitted_probability_floor_hits": int(np.sum(matrix <= epsilon)),
            "selected_probability_floor_hits": int(np.sum(selected <= epsilon)),
        }
        if oracle is not None:
            outcome_error = matrix - oracle[name]
            class_bias = outcome_error.mean(axis=0)
            row_kl = _oracle_kl_rows(oracle[name], matrix)
            outcome_record.update(
                {
                    "oracle_probability_mae": float(np.abs(outcome_error).mean()),
                    "oracle_probability_rmse": float(
                        np.sqrt(np.square(outcome_error).mean())
                    ),
                    "oracle_probability_mean_absolute_class_bias": float(
                        np.abs(class_bias).mean()
                    ),
                    "oracle_probability_rms_class_bias": float(
                        np.sqrt(np.square(class_bias).mean())
                    ),
                    "oracle_probability_maximum_absolute_class_bias": float(
                        np.abs(class_bias).max()
                    ),
                    "oracle_kl": float(row_kl.mean()),
                    "total_variation": float(
                        (0.5 * np.abs(outcome_error).sum(axis=1)).mean()
                    ),
                    "brier_regret": float(np.square(outcome_error).sum(axis=1).mean()),
                    "oracle_probability_floor_hits": int(
                        np.sum(oracle[name] <= epsilon)
                    ),
                    "positive_oracle_at_fitted_zero": int(
                        np.sum((oracle[name] > 0.0) & (matrix == 0.0))
                    ),
                }
            )
        outcome_records.append(outcome_record)

        for class_index, level in enumerate(levels):
            observed = observed_matrix[:, class_index]
            predicted = matrix[:, class_index]
            actual = oracle[name][:, class_index] if oracle is not None else None
            reliability = calibration_curve(
                observed,
                predicted,
                n_bins=n_bins,
                strategy=calibration_strategy,
                oracle_probability=actual,
            )
            binary_brier = float(np.square(predicted - observed).mean())
            prevalence = float(observed.mean())
            null_brier = prevalence * (1.0 - prevalence)
            auc, pr_auc = _binary_ranking_metrics(observed, predicted)
            ece = float(
                sum(item["count"] * abs(item["signed_gap"]) for item in reliability)
                / codes.shape[0]
            )
            class_record: Dict[str, Any] = {
                "outcome": name,
                "level": level,
                "class_index": class_index,
                "cardinality": len(levels),
                "n": int(codes.shape[0]),
                "event_count": int(observed.sum()),
                "observed_prevalence": prevalence,
                "mean_predicted": float(predicted.mean()),
                "calibration_in_the_large": float((observed - predicted).mean()),
                "ece": ece,
                "maximum_calibration_gap": float(
                    max(abs(item["signed_gap"]) for item in reliability)
                ),
                "auc": auc,
                "pr_auc": pr_auc,
                "auc_defined": auc is not None,
                "pr_auc_defined": pr_auc is not None,
                "binary_brier": binary_brier,
                "binary_brier_skill_vs_prevalence": (
                    float(1.0 - binary_brier / null_brier) if null_brier > 0.0 else None
                ),
                "binary_log_loss": _safe_binary_log_loss(observed, predicted, epsilon),
                "fitted_probability_floor_hits": int(np.sum(predicted <= epsilon)),
                "reliability_bins": reliability,
            }

            if oracle is not None:
                assert actual is not None
                error = predicted - actual
                class_record.update(
                    {
                        "oracle_probability_mae": float(np.abs(error).mean()),
                        "oracle_probability_rmse": float(
                            np.sqrt(np.square(error).mean())
                        ),
                        "oracle_probability_bias": float(error.mean()),
                        "oracle_probability_floor_hits": int(np.sum(actual <= epsilon)),
                    }
                )
                for band_index, (lower, upper) in enumerate(zip(bands, bands[1:])):
                    if band_index == len(bands) - 2:
                        in_band = np.logical_and(actual >= lower, actual <= upper)
                    else:
                        in_band = np.logical_and(actual >= lower, actual < upper)
                    if not np.any(in_band):
                        continue
                    band_error = error[in_band]
                    band_records.append(
                        {
                            "outcome": name,
                            "level": level,
                            "class_index": class_index,
                            "lower": lower,
                            "upper": upper,
                            "label": f"[{lower:g}, {upper:g}{']' if upper == 1.0 else ')'}",
                            "count": int(in_band.sum()),
                            "mean_oracle_probability": float(actual[in_band].mean()),
                            "mean_fitted_probability": float(predicted[in_band].mean()),
                            "bias": float(band_error.mean()),
                            "mae": float(np.abs(band_error).mean()),
                            "rmse": float(np.sqrt(np.square(band_error).mean())),
                        }
                    )
            class_records.append(class_record)

    all_auc = [record["auc"] for record in class_records if record["auc"] is not None]
    all_pr_auc = [
        record["pr_auc"] for record in class_records if record["pr_auc"] is not None
    ]
    partial_outcome_auc = []
    partial_outcome_pr_auc = []
    complete_outcome_auc = []
    complete_outcome_pr_auc = []
    outcome_ece = []
    for entry in normalised_schema:
        selected_classes = [
            record for record in class_records if record["outcome"] == entry["name"]
        ]
        auc_values = [
            record["auc"] for record in selected_classes if record["auc"] is not None
        ]
        pr_auc_values = [
            record["pr_auc"]
            for record in selected_classes
            if record["pr_auc"] is not None
        ]
        if auc_values:
            partial_outcome_auc.append(float(np.mean(auc_values)))
        if len(auc_values) == len(selected_classes):
            complete_outcome_auc.append(float(np.mean(auc_values)))
        if pr_auc_values:
            partial_outcome_pr_auc.append(float(np.mean(pr_auc_values)))
        if len(pr_auc_values) == len(selected_classes):
            complete_outcome_pr_auc.append(float(np.mean(pr_auc_values)))
        outcome_ece.append(
            float(np.mean([record["ece"] for record in selected_classes]))
        )
    required_ranking_classes = len(class_records)
    defined_auc_classes = len(all_auc)
    defined_pr_auc_classes = len(all_pr_auc)
    summary: Dict[str, Any] = {
        "n": int(codes.shape[0]),
        "n_outcomes": len(normalised_schema),
        "n_classes": len(class_records),
        # Primary aggregates give every semantic outcome equal weight; pooled
        # class-cell alternatives are retained under explicitly named keys.
        "macro_one_vs_rest_auc": (
            float(np.mean(complete_outcome_auc))
            if len(complete_outcome_auc) == len(normalised_schema)
            else None
        ),
        "macro_one_vs_rest_pr_auc": (
            float(np.mean(complete_outcome_pr_auc))
            if len(complete_outcome_pr_auc) == len(normalised_schema)
            else None
        ),
        "partial_macro_one_vs_rest_auc": (
            float(np.mean(partial_outcome_auc)) if partial_outcome_auc else None
        ),
        "partial_macro_one_vs_rest_pr_auc": (
            float(np.mean(partial_outcome_pr_auc)) if partial_outcome_pr_auc else None
        ),
        "pooled_class_macro_one_vs_rest_auc": (
            float(np.mean(all_auc))
            if defined_auc_classes == required_ranking_classes
            else None
        ),
        "pooled_class_macro_one_vs_rest_pr_auc": (
            float(np.mean(all_pr_auc))
            if defined_pr_auc_classes == required_ranking_classes
            else None
        ),
        "partial_pooled_class_macro_one_vs_rest_auc": (
            float(np.mean(all_auc)) if all_auc else None
        ),
        "partial_pooled_class_macro_one_vs_rest_pr_auc": (
            float(np.mean(all_pr_auc)) if all_pr_auc else None
        ),
        "one_vs_rest_auc_defined_classes": defined_auc_classes,
        "one_vs_rest_auc_required_classes": required_ranking_classes,
        "one_vs_rest_auc_coverage": float(
            defined_auc_classes / required_ranking_classes
        ),
        "one_vs_rest_pr_auc_defined_classes": defined_pr_auc_classes,
        "one_vs_rest_pr_auc_required_classes": required_ranking_classes,
        "one_vs_rest_pr_auc_coverage": float(
            defined_pr_auc_classes / required_ranking_classes
        ),
        "mean_outcome_nll": float(np.mean([item["nll"] for item in outcome_records])),
        "mean_outcome_multiclass_brier": float(
            np.mean([item["multiclass_brier"] for item in outcome_records])
        ),
        "mean_classwise_ece": float(np.mean(outcome_ece)),
        "maximum_classwise_calibration_gap": float(
            max(item["maximum_calibration_gap"] for item in class_records)
        ),
        "probability_floor_epsilon": float(epsilon),
        "fitted_probability_floor_hits": int(
            sum(item["fitted_probability_floor_hits"] for item in outcome_records)
        ),
        "selected_probability_floor_hits": int(
            sum(item["selected_probability_floor_hits"] for item in outcome_records)
        ),
    }
    if oracle is not None:
        stacked_error = np.concatenate(
            [
                fitted[entry["name"]] - oracle[entry["name"]]
                for entry in normalised_schema
            ],
            axis=1,
        )
        outcome_mae = np.array(
            [item["oracle_probability_mae"] for item in outcome_records]
        )
        outcome_mse = np.square(
            [item["oracle_probability_rmse"] for item in outcome_records]
        )
        outcome_mean_absolute_class_bias = np.array(
            [
                item["oracle_probability_mean_absolute_class_bias"]
                for item in outcome_records
            ]
        )
        outcome_rms_class_bias = np.array(
            [item["oracle_probability_rms_class_bias"] for item in outcome_records]
        )
        summary.update(
            {
                "oracle_probability_mae": float(outcome_mae.mean()),
                "oracle_probability_rmse": float(np.sqrt(outcome_mse.mean())),
                "oracle_probability_mean_absolute_class_bias": float(
                    outcome_mean_absolute_class_bias.mean()
                ),
                "oracle_probability_rms_class_bias": float(
                    np.sqrt(np.square(outcome_rms_class_bias).mean())
                ),
                "oracle_probability_maximum_absolute_class_bias": float(
                    max(
                        item["oracle_probability_maximum_absolute_class_bias"]
                        for item in outcome_records
                    )
                ),
                "mean_outcome_oracle_kl": float(
                    np.mean([item["oracle_kl"] for item in outcome_records])
                ),
                "mean_outcome_total_variation": float(
                    np.mean([item["total_variation"] for item in outcome_records])
                ),
                "mean_outcome_brier_regret": float(
                    np.mean([item["brier_regret"] for item in outcome_records])
                ),
                "pooled_class_cell_oracle_probability_mae": float(
                    np.abs(stacked_error).mean()
                ),
                "pooled_class_cell_oracle_probability_rmse": float(
                    np.sqrt(np.square(stacked_error).mean())
                ),
                "oracle_probability_floor_hits": int(
                    sum(
                        item["oracle_probability_floor_hits"]
                        for item in outcome_records
                    )
                ),
                "positive_oracle_at_fitted_zero": int(
                    sum(
                        item["positive_oracle_at_fitted_zero"]
                        for item in outcome_records
                    )
                ),
            }
        )

    return {
        "summary": summary,
        "outcomes": outcome_records,
        "classes": class_records,
        "true_probability_bands": band_records,
        "calibration": {
            "strategy": calibration_strategy,
            "requested_bins": n_bins,
        },
        "schema": [dict(entry) for entry in normalised_schema],
    }


def compare_probability_models(
    model_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    *,
    oracle_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
    true_probability_bands: Sequence[float] = DEFAULT_TRUE_PROBABILITY_BANDS,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Apply an identical scoring contract to any number of named models."""

    if not isinstance(model_probabilities, Mapping) or not model_probabilities:
        raise ValueError("model_probabilities must contain at least one named model.")
    results: Dict[str, Any] = {}
    for model_name, probabilities in model_probabilities.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Every model name must be a nonempty string.")
        results[model_name] = probability_diagnostics(
            probabilities,
            Y,
            schema,
            oracle_probabilities=oracle_probabilities,
            n_bins=n_bins,
            calibration_strategy=calibration_strategy,
            true_probability_bands=true_probability_bands,
            epsilon=epsilon,
        )
    first = next(iter(results.values()))
    return {
        "models": results,
        "schema": first["schema"],
        "calibration": first["calibration"],
        "has_oracle": oracle_probabilities is not None,
    }


def experiment_metric_record(
    diagnostics: Mapping[str, Any],
    model: str,
    factors: Mapping[str, Any],
    *,
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten one fitted run for multi-run trend plots."""

    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty string.")
    reserved = {"model", "scenario", "seed"}
    overlap = reserved.intersection(factors)
    if overlap:
        raise ValueError(f"factors use reserved names: {sorted(overlap)}.")
    summary = diagnostics["summary"]
    record: Dict[str, Any] = {
        "model": model,
        "scenario": scenario,
        "seed": seed,
        **dict(factors),
    }
    for key in (
        "macro_one_vs_rest_auc",
        "macro_one_vs_rest_pr_auc",
        "partial_macro_one_vs_rest_auc",
        "partial_macro_one_vs_rest_pr_auc",
        "one_vs_rest_auc_defined_classes",
        "one_vs_rest_auc_required_classes",
        "one_vs_rest_auc_coverage",
        "one_vs_rest_pr_auc_defined_classes",
        "one_vs_rest_pr_auc_required_classes",
        "one_vs_rest_pr_auc_coverage",
        "mean_outcome_nll",
        "mean_outcome_multiclass_brier",
        "mean_classwise_ece",
        "maximum_classwise_calibration_gap",
        "oracle_probability_mae",
        "oracle_probability_rmse",
        "oracle_probability_mean_absolute_class_bias",
        "oracle_probability_rms_class_bias",
        "oracle_probability_maximum_absolute_class_bias",
        "mean_outcome_oracle_kl",
        "mean_outcome_total_variation",
        "mean_outcome_brier_regret",
        "pooled_class_cell_oracle_probability_mae",
        "pooled_class_cell_oracle_probability_rmse",
        "fitted_probability_floor_hits",
        "selected_probability_floor_hits",
        "oracle_probability_floor_hits",
        "positive_oracle_at_fitted_zero",
    ):
        record[key] = summary.get(key)
    return record


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised without plot extra.
        raise RuntimeError(
            "Plotting requires Matplotlib; install this project with the 'plot' extra."
        ) from exc
    return plt


def _page_chunks(items: Sequence[Any], maximum: int) -> List[Sequence[Any]]:
    if maximum < 1:
        raise ValueError("maximum panels per page must be positive.")
    return [items[start : start + maximum] for start in range(0, len(items), maximum)]


def _panel_grid(count: int, columns: int = 3) -> Tuple[int, int]:
    columns = min(columns, count)
    rows = int(math.ceil(count / columns))
    return rows, columns


def _save_figure(fig: Any, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    return path


def _annotate_bin_counts(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    bins: Sequence[Mapping[str, Any]],
    *,
    color: Any,
) -> None:
    """Put the effective bin size next to every reliability point."""

    for x_value, y_value, item in zip(x, y, bins):
        axis.annotate(
            f"n={int(item['count'])}",
            (float(x_value), float(y_value)),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color=color,
            alpha=0.85,
        )


def _bin_count_summary(bins: Sequence[Mapping[str, Any]]) -> str:
    counts = [int(item["count"]) for item in bins]
    if not counts:
        return "no displayed bins"
    if min(counts) == max(counts):
        return f"bin n={counts[0]}"
    return f"bin n={min(counts)}-{max(counts)}"


def _display_model_label(model: Any) -> str:
    label = str(model).replace("_", " ")
    return "categorical CVAE" if label.lower() == "cvae" else label


def plot_calibration_pages(
    diagnostics: Mapping[str, Any],
    output_dir: Path,
    *,
    filename_prefix: str = "categorical",
    model_label: str = "CVAE",
    maximum_panels_per_page: int = 12,
    probability_limit: float = 1.0,
    dpi: int = 150,
) -> List[Path]:
    """Write paginated one-vs-rest observed calibration plots."""

    if not 0.0 < probability_limit <= 1.0:
        raise ValueError("probability_limit must lie in (0, 1].")
    plt = _pyplot()
    records = list(diagnostics["classes"])
    paths: List[Path] = []
    suffix = (
        "calibration" if probability_limit == 1.0 else "calibration_low_probability"
    )
    for page_index, page in enumerate(
        _page_chunks(records, maximum_panels_per_page), start=1
    ):
        n_rows, n_columns = _panel_grid(len(page))
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.2 * n_columns, 4.05 * n_rows),
            squeeze=False,
        )
        for axis, record in zip(axes.flat, page):
            bins = [
                item
                for item in record["reliability_bins"]
                if item["mean_predicted"] <= probability_limit
            ]
            axis.plot(
                [0.0, probability_limit],
                [0.0, probability_limit],
                linestyle="--",
                color="0.45",
                linewidth=1.0,
                label="perfect calibration",
            )
            if bins:
                x = np.array([item["mean_predicted"] for item in bins])
                y = np.array([item["observed_fraction"] for item in bins])
                lower = np.array([item["observed_wilson_lower_95"] for item in bins])
                upper = np.array([item["observed_wilson_upper_95"] for item in bins])
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack(
                        (
                            np.maximum(y - lower, 0.0),
                            np.maximum(upper - y, 0.0),
                        )
                    ),
                    color="#1769aa",
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                    label=model_label,
                )
                _annotate_bin_counts(axis, x, y, bins, color="#1769aa")
                if "mean_oracle_probability" in bins[0]:
                    axis.plot(
                        x,
                        [item["mean_oracle_probability"] for item in bins],
                        color="#d97904",
                        marker="D",
                        markersize=3.5,
                        linewidth=1.2,
                        label="DGP probability",
                    )
            axis.set_xlim(0.0, probability_limit)
            # A low-prediction zoom must still expose severe underprediction;
            # clipping observed/oracle values to the x-axis limit would hide it.
            axis.set_ylim(0.0, 1.0)
            if probability_limit == 1.0:
                axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.18)
            axis.set_title(
                f"{record['outcome']}: {record['level']}\n"
                f"events={record['event_count']}/{record['n']}; "
                f"ECE={record['ece']:.3f}",
                fontsize=9,
            )
            axis.set_xlabel("mean predicted probability")
            axis.set_ylabel("observed fraction / oracle probability")
            if bins:
                axis.legend(fontsize=7, loc="best")
        for axis in axes.flat[len(page) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"{model_label} one-vs-rest calibration"
            + (
                f" (predicted probability 0-{probability_limit:g}; y axis retained 0-1)"
                if probability_limit < 1.0
                else ""
            )
            + "; labels give bin n",
            fontsize=13,
            y=0.997,
        )
        fig.subplots_adjust(top=0.90, hspace=0.48, wspace=0.30)
        filename = f"{filename_prefix}_{suffix}_{page_index:02d}.png"
        paths.append(_save_figure(fig, Path(output_dir) / filename, dpi))
    return paths


def plot_comparative_calibration_pages(
    comparison: Mapping[str, Any],
    output_dir: Path,
    *,
    filename_prefix: str = "categorical_comparison",
    maximum_panels_per_page: int = 12,
    probability_limit: float = 1.0,
    dpi: int = 150,
) -> List[Path]:
    """Overlay observed reliability curves for named candidate models."""

    if not 0.0 < probability_limit <= 1.0:
        raise ValueError("probability_limit must lie in (0, 1].")
    model_results = comparison["models"]
    if not model_results:
        raise ValueError("comparison contains no models.")
    first = next(iter(model_results.values()))
    class_keys = [(record["outcome"], record["level"]) for record in first["classes"]]
    record_maps = {
        model: {
            (record["outcome"], record["level"]): record
            for record in diagnostics["classes"]
        }
        for model, diagnostics in model_results.items()
    }
    plt = _pyplot()
    colors = plt.get_cmap("tab10")
    suffix = (
        "calibration" if probability_limit == 1.0 else "calibration_low_probability"
    )
    paths: List[Path] = []
    for page_index, page in enumerate(
        _page_chunks(class_keys, maximum_panels_per_page), start=1
    ):
        n_rows, n_columns = _panel_grid(len(page))
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.3 * n_columns, 4.15 * n_rows),
            squeeze=False,
        )
        for axis, key in zip(axes.flat, page):
            axis.plot(
                [0.0, probability_limit],
                [0.0, probability_limit],
                linestyle="--",
                color="0.45",
                linewidth=1.0,
            )
            displayed_bins: List[Mapping[str, Any]] = []
            for model_index, (model, model_records) in enumerate(record_maps.items()):
                record = model_records[key]
                bins = [
                    item
                    for item in record["reliability_bins"]
                    if item["mean_predicted"] <= probability_limit
                ]
                if not bins:
                    continue
                displayed_bins.extend(bins)
                x = np.array([item["mean_predicted"] for item in bins])
                y = np.array([item["observed_fraction"] for item in bins])
                lower = np.array([item["observed_wilson_lower_95"] for item in bins])
                upper = np.array([item["observed_wilson_upper_95"] for item in bins])
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack(
                        (
                            np.maximum(y - lower, 0.0),
                            np.maximum(upper - y, 0.0),
                        )
                    ),
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    capsize=1.5,
                    color=colors(model_index % 10),
                    linestyle="none",
                    markerfacecolor="white",
                    label=f"{_display_model_label(model)}: observed",
                )
                if "mean_oracle_probability" in bins[0]:
                    axis.plot(
                        x,
                        [item["mean_oracle_probability"] for item in bins],
                        marker="D",
                        markersize=3.2,
                        linewidth=1.2,
                        color=colors(model_index % 10),
                        label=f"{_display_model_label(model)}: oracle",
                    )
            first_record = next(iter(record_maps.values()))[key]
            axis.set_title(
                f"{key[0]}: {key[1]}\n"
                f"events={first_record['event_count']}/{first_record['n']}; "
                f"{_bin_count_summary(displayed_bins)}",
                fontsize=9,
            )
            axis.set_xlim(0.0, probability_limit)
            axis.set_ylim(0.0, 1.0)
            if probability_limit == 1.0:
                axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.16)
            axis.set_xlabel("mean predicted probability")
            axis.set_ylabel("observed fraction / oracle probability")
        for axis in axes.flat[len(page) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.948),
                ncol=min(4, len(labels)),
                fontsize=8,
            )
        fig.suptitle(
            "One-vs-rest observed calibration by model"
            + (
                f" (predicted probability 0-{probability_limit:g}; y axis retained 0-1)"
                if probability_limit < 1.0
                else ""
            ),
            fontsize=13,
            y=0.997,
        )
        fig.subplots_adjust(top=0.81, hspace=0.50, wspace=0.30)
        filename = f"{filename_prefix}_{suffix}_{page_index:02d}.png"
        paths.append(_save_figure(fig, Path(output_dir) / filename, dpi))
    return paths


def _aggregate_band_errors(
    diagnostics: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    # First pool class cells within each semantic outcome, then give each
    # represented outcome equal weight. This prevents a K=20 outcome from
    # automatically dominating a K=2 outcome solely because it has more cells.
    by_outcome: Dict[Tuple[str, str], Dict[str, float]] = {}
    for record in diagnostics["true_probability_bands"]:
        label = record["label"]
        key = (label, record["outcome"])
        target = by_outcome.setdefault(
            key,
            {
                "count": 0.0,
                "weighted_bias": 0.0,
                "weighted_mae": 0.0,
                "weighted_mse": 0.0,
                "lower": float(record["lower"]),
            },
        )
        count = float(record["count"])
        target["count"] += count
        target["weighted_bias"] += count * float(record["bias"])
        target["weighted_mae"] += count * float(record["mae"])
        target["weighted_mse"] += count * float(record["rmse"]) ** 2
    outcome_summaries: Dict[str, List[Dict[str, float]]] = {}
    for (label, _), item in by_outcome.items():
        count = item["count"]
        outcome_summaries.setdefault(label, []).append(
            {
                "count": count,
                "lower": item["lower"],
                "bias": item["weighted_bias"] / count,
                "mae": item["weighted_mae"] / count,
                "mse": item["weighted_mse"] / count,
            }
        )
    result: Dict[str, Dict[str, float]] = {}
    for label, outcome_items in outcome_summaries.items():
        result[label] = {
            "count": float(sum(item["count"] for item in outcome_items)),
            "n_outcomes": float(len(outcome_items)),
            "lower": outcome_items[0]["lower"],
            "bias": float(np.mean([item["bias"] for item in outcome_items])),
            "mae": float(np.mean([item["mae"] for item in outcome_items])),
            "rmse": float(math.sqrt(np.mean([item["mse"] for item in outcome_items]))),
        }
    return result


def plot_error_by_true_probability_band(
    comparison: Mapping[str, Any],
    output_path: Path,
    *,
    dpi: int = 150,
) -> Path:
    """Compare model probability errors within oracle-probability bands."""

    if not comparison.get("has_oracle", False):
        raise ValueError("True-probability-band plots require oracle probabilities.")
    model_results = comparison["models"]
    aggregates = {
        model: _aggregate_band_errors(diagnostics)
        for model, diagnostics in model_results.items()
    }
    all_labels = sorted(
        {label for values in aggregates.values() for label in values},
        key=lambda label: next(
            values[label]["lower"] for values in aggregates.values() if label in values
        ),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), squeeze=False)
    colors = plt.get_cmap("tab10")
    x = np.arange(len(all_labels))
    for model_index, (model, values) in enumerate(aggregates.items()):
        for axis, metric in zip(axes.flat, ("bias", "mae", "rmse")):
            y = np.array(
                [
                    values[label][metric] if label in values else np.nan
                    for label in all_labels
                ]
            )
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1.5,
                color=colors(model_index % 10),
                label=model,
            )
    titles = {
        "bias": "signed error (fitted - oracle)",
        "mae": "mean absolute error",
        "rmse": "root mean squared error",
    }
    for axis, metric in zip(axes.flat, ("bias", "mae", "rmse")):
        if metric == "bias":
            axis.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
        axis.set_xticks(x)
        axis.set_xticklabels(all_labels, rotation=35, ha="right")
        axis.set_xlabel("oracle conditional-probability range")
        axis.set_title(titles[metric], fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle(
        "Probability error by true probability (equal weight per outcome)", y=1.03
    )
    return _save_figure(fig, Path(output_path), dpi)


def plot_oracle_agreement_pages(
    probabilities: Mapping[str, np.ndarray],
    oracle_probabilities: Mapping[str, np.ndarray],
    schema: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    filename_prefix: str = "categorical",
    model_label: str = "CVAE",
    maximum_panels_per_page: int = 12,
    probability_limit: float = 1.0,
    dpi: int = 150,
) -> List[Path]:
    """Plot row-level fitted probability against the known DGP probability."""

    if not 0.0 < probability_limit <= 1.0:
        raise ValueError("probability_limit must lie in (0, 1].")
    normalised_schema = _normalise_schema(schema)
    first_name = normalised_schema[0]["name"]
    n_rows = np.asarray(probabilities[first_name]).shape[0]
    fitted = _validate_probability_mapping(
        probabilities, n_rows, normalised_schema, "probabilities"
    )
    oracle = _validate_probability_mapping(
        oracle_probabilities, n_rows, normalised_schema, "oracle_probabilities"
    )
    plt = _pyplot()
    records = [
        (entry, class_index, level)
        for entry in normalised_schema
        for class_index, level in enumerate(entry["levels"])
    ]
    paths: List[Path] = []
    suffix = (
        "oracle_agreement"
        if probability_limit == 1.0
        else "oracle_agreement_low_probability"
    )
    for page_index, page in enumerate(
        _page_chunks(records, maximum_panels_per_page), start=1
    ):
        plot_rows, plot_columns = _panel_grid(len(page))
        fig, axes = plt.subplots(
            plot_rows,
            plot_columns,
            figsize=(4.0 * plot_columns, 3.45 * plot_rows),
            squeeze=False,
        )
        for axis, (entry, class_index, level) in zip(axes.flat, page):
            name = entry["name"]
            actual = oracle[name][:, class_index]
            predicted = fitted[name][:, class_index]
            show = actual <= probability_limit
            x = actual[show]
            y = predicted[show]
            if x.size >= 100:
                axis.hexbin(
                    x,
                    y,
                    gridsize=32,
                    extent=(0.0, probability_limit, 0.0, 1.0),
                    mincnt=1,
                    cmap="Blues",
                )
            else:
                axis.scatter(
                    x,
                    y,
                    s=10,
                    alpha=0.5,
                    color="#1769aa",
                    rasterized=True,
                )
            axis.plot(
                [0.0, probability_limit],
                [0.0, probability_limit],
                linestyle="--",
                color="#a83232",
                linewidth=1.0,
            )
            axis.set_xlim(0.0, probability_limit)
            axis.set_ylim(0.0, 1.0)
            if probability_limit == 1.0:
                axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.12)
            if x.size:
                shown_error = y - x
                detail = (
                    f"shown n={x.size}; MAE={np.abs(shown_error).mean():.3f}; "
                    f"RMSE={np.sqrt(np.square(shown_error).mean()):.3f}"
                )
            else:
                detail = "shown n=0"
            axis.set_title(f"{name}: {level}\n{detail}", fontsize=9)
            axis.set_xlabel("oracle conditional probability")
            axis.set_ylabel(f"{model_label} probability")
        for axis in axes.flat[len(page) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"{model_label} versus data-generating conditional probabilities"
            + (
                f" (oracle <= {probability_limit:g})" if probability_limit < 1.0 else ""
            ),
            fontsize=13,
        )
        filename = f"{filename_prefix}_{suffix}_{page_index:02d}.png"
        paths.append(_save_figure(fig, Path(output_dir) / filename, dpi))
    return paths


def plot_metric_panel(
    diagnostics: Mapping[str, Any],
    output_path: Path,
    *,
    model_label: str = "CVAE",
    dpi: int = 150,
) -> Path:
    """Write a compact per-class panel of discrimination and probability error."""

    plt = _pyplot()
    records = list(diagnostics["classes"])
    has_oracle = all("oracle_probability_rmse" in item for item in records)
    metric_columns = [
        ("auc", "one-vs-rest AUC", (0.0, 1.0), 0.5),
        ("pr_auc", "one-vs-rest PR-AUC", (0.0, 1.0), None),
        ("binary_brier", "binary Brier", (0.0, None), None),
        ("ece", "calibration ECE", (0.0, None), 0.0),
    ]
    if has_oracle:
        metric_columns.append(
            ("oracle_probability_rmse", "oracle probability RMSE", (0.0, None), 0.0)
        )
    height = max(3.5, 0.30 * len(records) + 1.7)
    fig, axes = plt.subplots(
        1,
        len(metric_columns),
        figsize=(3.25 * len(metric_columns), height),
        sharey=True,
        squeeze=False,
    )
    labels = [f"{item['outcome']}: {item['level']}" for item in records]
    positions = np.arange(len(records))[::-1]
    for column_index, (key, title, limits, reference) in enumerate(metric_columns):
        axis = axes[0, column_index]
        values = np.array(
            [np.nan if item[key] is None else item[key] for item in records],
            dtype=np.float64,
        )
        axis.scatter(values, positions, color="#1769aa", s=22)
        if reference is not None:
            axis.axvline(reference, color="0.5", linestyle="--", linewidth=1.0)
        right = limits[1]
        if right is None:
            finite = values[np.isfinite(values)]
            right = max(float(finite.max()) * 1.12, 0.01) if finite.size else 1.0
        axis.set_xlim(limits[0], right)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=0.2)
        axis.set_yticks(positions)
        if column_index == 0:
            axis.set_yticklabels(labels, fontsize=8)
        else:
            axis.tick_params(axis="y", labelleft=False)
    fig.suptitle(f"{model_label} marginal probability metrics by class", fontsize=13)
    return _save_figure(fig, Path(output_path), dpi)


def _factor_order(values: Sequence[Any]) -> List[Any]:
    unique = list(dict.fromkeys(values))
    try:
        return sorted(unique, key=float)
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def plot_metric_trends(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    factor_labels: Mapping[str, str],
    filename_prefix: str = "categorical",
    metrics: Sequence[Tuple[str, str]] = (
        ("macro_one_vs_rest_auc", "macro ROC-AUC"),
        ("macro_one_vs_rest_pr_auc", "macro PR-AUC"),
        ("mean_outcome_multiclass_brier", "mean multiclass Brier"),
        ("mean_classwise_ece", "mean classwise ECE"),
        ("oracle_probability_rmse", "equal-outcome probability RMSE"),
        ("mean_outcome_total_variation", "equal-outcome total variation"),
        ("mean_outcome_oracle_kl", "equal-outcome oracle KL"),
        ("mean_outcome_brier_regret", "equal-outcome Brier regret"),
    ),
    dpi: int = 150,
) -> List[Path]:
    """Plot raw fitted-run metrics and descriptive cell means by factor.

    Each input row should be produced by :func:`experiment_metric_record` (or
    obey the same flat contract).  Values at a factor level are marginal means
    across the supplied rows; a factorial experiment should therefore be
    balanced if these are to be interpreted as isolated factor effects.  No
    inferential error bars are drawn: initialization replicates, paired models,
    and crossed scenarios are generally correlated and require a clustered
    analysis in the protocol-level report.
    """

    if not records:
        raise ValueError("records must contain at least one experiment result.")
    if not factor_labels:
        raise ValueError("factor_labels must contain at least one factor.")
    models = list(dict.fromkeys(str(record["model"]) for record in records))
    plt = _pyplot()
    colors = plt.get_cmap("tab10")
    paths: List[Path] = []
    output_dir = Path(output_dir)

    for factor, factor_label in factor_labels.items():
        if any(factor not in record for record in records):
            raise ValueError(f"Not every experiment record contains factor {factor!r}.")
        factor_values = _factor_order([record[factor] for record in records])
        n_columns = 2
        n_rows = int(math.ceil(len(metrics) / n_columns))
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(6.0 * n_columns, 3.35 * n_rows),
            squeeze=False,
        )
        x = np.arange(len(factor_values))
        for axis, (metric, metric_label) in zip(axes.flat, metrics):
            for model_index, model in enumerate(models):
                means = []
                offset = (model_index - (len(models) - 1) / 2.0) * (
                    0.45 / max(len(models), 1)
                )
                for factor_index, factor_value in enumerate(factor_values):
                    selected = [
                        record.get(metric)
                        for record in records
                        if str(record["model"]) == model
                        and record[factor] == factor_value
                        and record.get(metric) is not None
                    ]
                    numeric = np.asarray(selected, dtype=np.float64)
                    means.append(float(numeric.mean()) if numeric.size else np.nan)
                    if numeric.size:
                        axis.scatter(
                            np.repeat(x[factor_index] + offset, numeric.size),
                            numeric,
                            s=13,
                            alpha=0.28,
                            color=colors(model_index % 10),
                            linewidths=0.0,
                        )
                axis.plot(
                    x + offset,
                    means,
                    marker="o",
                    linewidth=1.5,
                    color=colors(model_index % 10),
                    label=f"{model} cell mean",
                )
            axis.set_title(metric_label)
            axis.set_xticks(x)
            axis.set_xticklabels([str(value) for value in factor_values])
            axis.set_xlabel(factor_label)
            axis.grid(axis="y", alpha=0.2)
        for axis in axes.flat[len(metrics) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels))
        fig.suptitle(
            f"Marginal conditional-probability performance by {factor_label}\n"
            "dots are fitted runs; lines are descriptive cell means",
            fontsize=13,
            y=1.03,
        )
        safe_factor = re.sub(r"[^A-Za-z0-9_.-]+", "_", factor).strip("_")
        path = output_dir / f"{filename_prefix}_trend_{safe_factor}.png"
        paths.append(_save_figure(fig, path, dpi))
    return paths


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _escape_markdown_table(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _markdown_image_link(path: Path, markdown_parent: Path, caption: str) -> str:
    relative = Path(os.path.relpath(path, markdown_parent)).as_posix()
    safe_caption = (
        str(caption).replace("\r", " ").replace("\n", " ").replace("]", "\\]")
    )
    target = (
        f"<{relative}>"
        if any(character.isspace() for character in relative)
        else relative
    )
    return f"![{safe_caption}]({target})"


def render_probability_diagnostics_markdown(
    diagnostics: Mapping[str, Any],
    image_paths: Sequence[Tuple[Path, str]],
    *,
    markdown_parent: Path,
    title: str = "Conditional-probability diagnostics",
) -> str:
    """Render a self-contained Markdown section linking generated images."""

    summary = diagnostics["summary"]
    lines = [
        f"# {str(title).replace(chr(10), ' ').replace(chr(13), ' ')}",
        "",
        (
            "Reliability plots compare predictions with observed event rates in "
            "groups. Oracle-agreement plots are simulation-only and compare each "
            "row/class prediction directly with its known data-generating "
            "conditional probability."
        ),
        "",
        "## Overall metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| test rows | {summary['n']} |",
        f"| semantic outcomes | {summary['n_outcomes']} |",
        f"| one-vs-rest classes | {summary['n_classes']} |",
        f"| complete equal-outcome macro one-vs-rest AUC | {_format_number(summary['macro_one_vs_rest_auc'])} |",
        f"| AUC class coverage | {summary['one_vs_rest_auc_defined_classes']} / {summary['one_vs_rest_auc_required_classes']} |",
        f"| partial macro one-vs-rest AUC (defined classes only) | {_format_number(summary['partial_macro_one_vs_rest_auc'])} |",
        f"| complete equal-outcome macro one-vs-rest PR-AUC | {_format_number(summary['macro_one_vs_rest_pr_auc'])} |",
        f"| PR-AUC class coverage | {summary['one_vs_rest_pr_auc_defined_classes']} / {summary['one_vs_rest_pr_auc_required_classes']} |",
        f"| partial macro one-vs-rest PR-AUC (defined classes only) | {_format_number(summary['partial_macro_one_vs_rest_pr_auc'])} |",
        f"| mean outcome NLL | {_format_number(summary['mean_outcome_nll'])} |",
        (
            "| mean multiclass Brier (sum over levels) | "
            f"{_format_number(summary['mean_outcome_multiclass_brier'])} |"
        ),
        f"| mean classwise ECE | {_format_number(summary['mean_classwise_ece'])} |",
        (
            "| maximum classwise calibration gap | "
            f"{_format_number(summary['maximum_classwise_calibration_gap'])} |"
        ),
        f"| fitted probability floor hits (<= {summary['probability_floor_epsilon']:.1e}) | {summary['fitted_probability_floor_hits']} |",
        f"| observed-category probability floor hits | {summary['selected_probability_floor_hits']} |",
    ]
    if "oracle_probability_rmse" in summary:
        lines.extend(
            [
                f"| equal-outcome probability MAE vs oracle | {_format_number(summary['oracle_probability_mae'])} |",
                f"| equal-outcome probability RMSE vs oracle | {_format_number(summary['oracle_probability_rmse'])} |",
                f"| equal-outcome mean absolute class bias vs oracle | {_format_number(summary['oracle_probability_mean_absolute_class_bias'])} |",
                f"| equal-outcome RMS class bias vs oracle | {_format_number(summary['oracle_probability_rms_class_bias'])} |",
                f"| maximum absolute class bias vs oracle | {_format_number(summary['oracle_probability_maximum_absolute_class_bias'])} |",
                f"| equal-outcome total variation vs oracle | {_format_number(summary['mean_outcome_total_variation'])} |",
                f"| equal-outcome KL from oracle | {_format_number(summary['mean_outcome_oracle_kl'])} |",
                f"| equal-outcome expected Brier regret | {_format_number(summary['mean_outcome_brier_regret'])} |",
                f"| oracle probability floor hits | {summary['oracle_probability_floor_hits']} |",
                f"| positive-oracle cells assigned fitted zero | {summary['positive_oracle_at_fitted_zero']} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-class metrics",
            "",
            "| outcome | level | events / n | prevalence | mean predicted | ROC-AUC | PR-AUC | binary Brier | ECE | oracle MAE | oracle RMSE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in diagnostics["classes"]:
        lines.append(
            f"| {_escape_markdown_table(record['outcome'])} | "
            f"{_escape_markdown_table(record['level'])} | "
            f"{record['event_count']} / {record['n']} | "
            f"{_format_number(record['observed_prevalence'])} | "
            f"{_format_number(record['mean_predicted'])} | "
            f"{_format_number(record['auc'])} | "
            f"{_format_number(record['pr_auc'])} | "
            f"{_format_number(record['binary_brier'])} | "
            f"{_format_number(record['ece'])} | "
            f"{_format_number(record.get('oracle_probability_mae'))} | "
            f"{_format_number(record.get('oracle_probability_rmse'))} |"
        )

    if diagnostics["true_probability_bands"]:
        lines.extend(
            [
                "",
                "## Error by true conditional-probability range",
                "",
                "| outcome | level | true-probability range | row/classes | mean oracle | mean fitted | bias | MAE | RMSE |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in diagnostics["true_probability_bands"]:
            lines.append(
                f"| {_escape_markdown_table(record['outcome'])} | "
                f"{_escape_markdown_table(record['level'])} | "
                f"{_escape_markdown_table(record['label'])} | "
                f"{record['count']} | {_format_number(record['mean_oracle_probability'])} | "
                f"{_format_number(record['mean_fitted_probability'])} | "
                f"{_format_number(record['bias'])} | {_format_number(record['mae'])} | "
                f"{_format_number(record['rmse'])} |"
            )

    if image_paths:
        lines.extend(["", "## Figures", ""])
        for path, caption in image_paths:
            lines.extend(
                [
                    f"### {str(caption).replace(chr(10), ' ').replace(chr(13), ' ')}",
                    "",
                    _markdown_image_link(Path(path), Path(markdown_parent), caption),
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- ROC-AUC and PR-AUC measure ranking, not calibration; high values can coexist with badly biased probabilities.",
            "- PR-AUC is non-interpolated average precision. Its prevalence-null reference is the observed class prevalence.",
            "- A binary Brier score depends strongly on prevalence. Compare like-for-like classes or use the prevalence-null skill score in machine-readable diagnostics.",
            "- Multiclass Brier here is the sum of squared level errors per outcome, then averaged across rows.",
            "- ECE depends on the declared binning rule and should be read with the plotted bin counts and Wilson intervals.",
            "- In simulation, each fitted-probability bin also shows its mean oracle probability; this separates model bias from binomial noise in the observed fraction.",
            "- AUC is `NA` when a test sample contains no observed event or no observed non-event for a class.",
            "- Complete macro AUC/PR-AUC is `NA` unless every required one-vs-rest class is estimable; partial macros are explicitly labeled and must not be treated as complete evidence.",
            "- Reliability-plot point labels give the effective bin sample size. Low-prediction zooms retain the full observed/oracle y-axis so severe underprediction is not clipped.",
            "- Oracle KL is exact: a fitted zero where the oracle assigns positive mass yields infinite regret. Floor-hit counts disclose where clipped realized log scores need caution.",
            "",
        ]
    )
    return "\n".join(lines)


def render_probability_comparison_markdown(
    comparison: Mapping[str, Any],
    image_paths: Sequence[Tuple[Path, str]],
    *,
    markdown_parent: Path,
    title: str = "Conditional-probability model comparison",
) -> str:
    """Render a compact multi-model Markdown comparison and figure index."""

    lines = [
        f"# {str(title).replace(chr(10), ' ').replace(chr(13), ' ')}",
        "",
        (
            "Observed reliability and known-oracle agreement answer different "
            "questions. The former estimates calibration from realized events; "
            "the latter directly measures row-level conditional-probability "
            "error and is available only because this is a simulation."
        ),
        "",
        "## Overall metrics by model",
        "",
        "| model | complete macro ROC-AUC | AUC coverage | partial ROC-AUC | complete macro PR-AUC | PR coverage | partial PR-AUC | outcome NLL | multiclass Brier | classwise ECE | oracle MAE | oracle RMSE | max class bias | total variation | oracle KL | Brier regret | fitted floor hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, diagnostics in comparison["models"].items():
        summary = diagnostics["summary"]
        lines.append(
            f"| {_escape_markdown_table(model)} | "
            f"{_format_number(summary['macro_one_vs_rest_auc'])} | "
            f"{summary['one_vs_rest_auc_defined_classes']} / {summary['one_vs_rest_auc_required_classes']} | "
            f"{_format_number(summary['partial_macro_one_vs_rest_auc'])} | "
            f"{_format_number(summary['macro_one_vs_rest_pr_auc'])} | "
            f"{summary['one_vs_rest_pr_auc_defined_classes']} / {summary['one_vs_rest_pr_auc_required_classes']} | "
            f"{_format_number(summary['partial_macro_one_vs_rest_pr_auc'])} | "
            f"{_format_number(summary['mean_outcome_nll'])} | "
            f"{_format_number(summary['mean_outcome_multiclass_brier'])} | "
            f"{_format_number(summary['mean_classwise_ece'])} | "
            f"{_format_number(summary.get('oracle_probability_mae'))} | "
            f"{_format_number(summary.get('oracle_probability_rmse'))} | "
            f"{_format_number(summary.get('oracle_probability_maximum_absolute_class_bias'))} | "
            f"{_format_number(summary.get('mean_outcome_total_variation'))} | "
            f"{_format_number(summary.get('mean_outcome_oracle_kl'))} | "
            f"{_format_number(summary.get('mean_outcome_brier_regret'))} | "
            f"{summary['fitted_probability_floor_hits']} |"
        )
    lines.extend(
        [
            "",
            "## Per-class metrics by model",
            "",
            "| model | outcome | level | events / n | prevalence | mean predicted | ROC-AUC | PR-AUC | binary Brier | ECE | oracle RMSE |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, diagnostics in comparison["models"].items():
        for record in diagnostics["classes"]:
            lines.append(
                f"| {_escape_markdown_table(model)} | "
                f"{_escape_markdown_table(record['outcome'])} | "
                f"{_escape_markdown_table(record['level'])} | "
                f"{record['event_count']} / {record['n']} | "
                f"{_format_number(record['observed_prevalence'])} | "
                f"{_format_number(record['mean_predicted'])} | "
                f"{_format_number(record['auc'])} | "
                f"{_format_number(record['pr_auc'])} | "
                f"{_format_number(record['binary_brier'])} | "
                f"{_format_number(record['ece'])} | "
                f"{_format_number(record.get('oracle_probability_rmse'))} |"
            )
    if image_paths:
        lines.extend(["", "## Figures", ""])
        for path, caption in image_paths:
            lines.extend(
                [
                    f"### {str(caption).replace(chr(10), ' ').replace(chr(13), ' ')}",
                    "",
                    _markdown_image_link(Path(path), Path(markdown_parent), caption),
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- ROC-AUC and PR-AUC assess discrimination, not probability calibration.",
            "- PR-AUC is non-interpolated average precision; its prevalence-null reference differs by class.",
            "- Reliability points are bin averages with 95% Wilson intervals for observed event fractions.",
            "- Simulation reliability panels also connect the mean oracle probability within each candidate model's fitted-probability bins.",
            "- Oracle MAE/RMSE compare every fitted probability with the exact data-generating probability for the same row and class.",
            "- Multiclass Brier is summed over levels within each semantic outcome; binary Brier is one class versus the rest.",
            "- Complete macro AUC/PR-AUC is `NA` unless every required class is estimable. Defined-class partial macros and coverage are shown separately.",
            "- Reliability points are labeled with bin n, and low-prediction zooms retain the full y-axis to expose underprediction.",
            "- Oracle KL is exact; unsupported positive oracle mass produces infinite regret. Fitted floor hits are shown explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def write_probability_diagnostic_bundle(
    output_dir: Path,
    filename_prefix: str,
    probabilities: Mapping[str, np.ndarray],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    *,
    oracle_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
    true_probability_bands: Sequence[float] = DEFAULT_TRUE_PROBABILITY_BANDS,
    low_probability_limit: Optional[float] = 0.20,
    maximum_panels_per_page: int = 12,
    model_label: str = "CVAE",
    title: str = "Conditional-probability diagnostics",
    dpi: int = 150,
) -> Dict[str, Any]:
    """Compute metrics and write PNG figures plus a Markdown report fragment."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = probability_diagnostics(
        probabilities,
        Y,
        schema,
        oracle_probabilities=oracle_probabilities,
        n_bins=n_bins,
        calibration_strategy=calibration_strategy,
        true_probability_bands=true_probability_bands,
    )
    image_paths: List[Tuple[Path, str]] = []
    metric_path = plot_metric_panel(
        diagnostics,
        output_dir / f"{filename_prefix}_metric_panel.png",
        model_label=model_label,
        dpi=dpi,
    )
    image_paths.append((metric_path, "Per-class metric panel"))
    for path in plot_calibration_pages(
        diagnostics,
        output_dir,
        filename_prefix=filename_prefix,
        model_label=model_label,
        maximum_panels_per_page=maximum_panels_per_page,
        dpi=dpi,
    ):
        image_paths.append((path, "Observed calibration"))
    if low_probability_limit is not None:
        for path in plot_calibration_pages(
            diagnostics,
            output_dir,
            filename_prefix=filename_prefix,
            model_label=model_label,
            maximum_panels_per_page=maximum_panels_per_page,
            probability_limit=float(low_probability_limit),
            dpi=dpi,
        ):
            image_paths.append((path, "Observed calibration: low-probability zoom"))
    if oracle_probabilities is not None:
        for path in plot_oracle_agreement_pages(
            probabilities,
            oracle_probabilities,
            schema,
            output_dir,
            filename_prefix=filename_prefix,
            model_label=model_label,
            maximum_panels_per_page=maximum_panels_per_page,
            dpi=dpi,
        ):
            image_paths.append((path, "Fitted versus oracle probabilities"))
        if low_probability_limit is not None:
            for path in plot_oracle_agreement_pages(
                probabilities,
                oracle_probabilities,
                schema,
                output_dir,
                filename_prefix=filename_prefix,
                model_label=model_label,
                maximum_panels_per_page=maximum_panels_per_page,
                probability_limit=float(low_probability_limit),
                dpi=dpi,
            ):
                image_paths.append(
                    (path, "Fitted versus oracle probabilities: low-probability zoom")
                )

    markdown_path = output_dir / f"{filename_prefix}_probability_diagnostics.md"
    markdown_path.write_text(
        render_probability_diagnostics_markdown(
            diagnostics,
            image_paths,
            markdown_parent=markdown_path.parent,
            title=title,
        ),
        encoding="utf-8",
    )
    return {
        "diagnostics": diagnostics,
        "image_paths": [str(path) for path, _ in image_paths],
        "markdown_path": str(markdown_path),
    }


def write_probability_comparison_bundle(
    output_dir: Path,
    filename_prefix: str,
    model_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    Y: np.ndarray,
    schema: Sequence[Mapping[str, Any]],
    *,
    oracle_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
    true_probability_bands: Sequence[float] = DEFAULT_TRUE_PROBABILITY_BANDS,
    low_probability_limit: Optional[float] = 0.20,
    maximum_panels_per_page: int = 12,
    title: str = "Conditional-probability model comparison",
    dpi: int = 150,
) -> Dict[str, Any]:
    """Write comparative metrics, PNGs, and a Markdown report fragment."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_probability_models(
        model_probabilities,
        Y,
        schema,
        oracle_probabilities=oracle_probabilities,
        n_bins=n_bins,
        calibration_strategy=calibration_strategy,
        true_probability_bands=true_probability_bands,
    )
    image_paths: List[Tuple[Path, str]] = []
    for path in plot_comparative_calibration_pages(
        comparison,
        output_dir,
        filename_prefix=filename_prefix,
        maximum_panels_per_page=maximum_panels_per_page,
        dpi=dpi,
    ):
        image_paths.append((path, "Observed calibration by model"))
    if low_probability_limit is not None:
        for path in plot_comparative_calibration_pages(
            comparison,
            output_dir,
            filename_prefix=filename_prefix,
            maximum_panels_per_page=maximum_panels_per_page,
            probability_limit=float(low_probability_limit),
            dpi=dpi,
        ):
            image_paths.append(
                (path, "Observed calibration by model: low-probability zoom")
            )

    used_slugs = set()
    for model_index, (model, probabilities) in enumerate(model_probabilities.items()):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
        if slug in used_slugs:
            slug = f"{slug}_{model_index + 1}"
        used_slugs.add(slug)
        metric_path = plot_metric_panel(
            comparison["models"][model],
            output_dir / f"{filename_prefix}_{slug}_metric_panel.png",
            model_label=model,
            dpi=dpi,
        )
        image_paths.append((metric_path, f"{model}: per-class metric panel"))
        if oracle_probabilities is not None:
            for path in plot_oracle_agreement_pages(
                probabilities,
                oracle_probabilities,
                schema,
                output_dir,
                filename_prefix=f"{filename_prefix}_{slug}",
                model_label=model,
                maximum_panels_per_page=maximum_panels_per_page,
                dpi=dpi,
            ):
                image_paths.append(
                    (path, f"{model}: fitted versus oracle probabilities")
                )
            if low_probability_limit is not None:
                for path in plot_oracle_agreement_pages(
                    probabilities,
                    oracle_probabilities,
                    schema,
                    output_dir,
                    filename_prefix=f"{filename_prefix}_{slug}",
                    model_label=model,
                    maximum_panels_per_page=maximum_panels_per_page,
                    probability_limit=float(low_probability_limit),
                    dpi=dpi,
                ):
                    image_paths.append(
                        (
                            path,
                            f"{model}: fitted versus oracle probabilities, low-probability zoom",
                        )
                    )
    if oracle_probabilities is not None:
        band_path = plot_error_by_true_probability_band(
            comparison,
            output_dir / f"{filename_prefix}_true_probability_band_errors.png",
            dpi=dpi,
        )
        image_paths.append((band_path, "Probability error by true-probability band"))

    markdown_path = output_dir / f"{filename_prefix}_probability_comparison.md"
    markdown_path.write_text(
        render_probability_comparison_markdown(
            comparison,
            image_paths,
            markdown_parent=markdown_path.parent,
            title=title,
        ),
        encoding="utf-8",
    )
    return {
        "comparison": comparison,
        "image_paths": [str(path) for path, _ in image_paths],
        "markdown_path": str(markdown_path),
    }


__all__ = [
    "DEFAULT_TRUE_PROBABILITY_BANDS",
    "binary_auc",
    "binary_pr_auc",
    "calibration_curve",
    "compare_probability_models",
    "experiment_metric_record",
    "plot_calibration_pages",
    "plot_comparative_calibration_pages",
    "plot_error_by_true_probability_band",
    "plot_metric_panel",
    "plot_metric_trends",
    "plot_oracle_agreement_pages",
    "probability_diagnostics",
    "render_probability_comparison_markdown",
    "render_probability_diagnostics_markdown",
    "write_probability_comparison_bundle",
    "write_probability_diagnostic_bundle",
]
