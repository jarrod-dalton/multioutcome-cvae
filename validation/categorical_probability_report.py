"""Render the categorical conditional-probability experiment.

The renderer is intentionally separate from fitting.  It consumes either the
in-memory mapping returned by ``run_probability_validation()`` or its portable
JSON representation and writes a Markdown report plus deterministic PNGs.

Example::

    python -m validation.categorical_probability_report \
        validation-results.json \
        --output docs/categorical_probability_validation.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from validation.probability_diagnostics import plot_comparative_calibration_pages


REPORT_VERSION = "categorical-conditional-probability-report-v1"
CANDIDATE_MODELS: Tuple[str, ...] = ("cvae", "independent_softmax")
MODEL_LABELS = {
    "cvae": "categorical CVAE",
    "independent_softmax": "independent softmax",
    "intercept_null": "intercept-only null",
    "oracle": "DGP oracle",
}
MODEL_COLORS = {
    "cvae": "#1769aa",
    "independent_softmax": "#d97904",
    "intercept_null": "#777777",
    "oracle": "#2b8a3e",
}


MetricGetter = Callable[[Mapping[str, Any], str], Optional[float]]


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "Report plotting requires Matplotlib; install the project's test or plot extra."
        ) from exc
    return plt


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _t_critical_95(degrees_of_freedom: int) -> float:
    # Two-sided 95% Student-t critical values. Values above 30 use the normal
    # approximation; canonical cell inference has df=4.
    table = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
        2.048, 2.045, 2.042,
    )
    if degrees_of_freedom < 1:
        return 0.0
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    return 1.96


def _mean_95_half_width(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("Metric groups must contain finite values.")
    standard_error = (
        float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    )
    half_width = _t_critical_95(array.size - 1) * standard_error
    return float(array.mean()), half_width


def _save_figure(fig: Any, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    return path.resolve()


def _markdown_image(path: Path, report_parent: Path, caption: str) -> str:
    relative = Path(os.path.relpath(path, report_parent)).as_posix()
    return f"![{caption}]({relative})"


def validate_probability_run(run: Mapping[str, Any]) -> None:
    """Validate the stable report-facing subset of the runner contract."""

    required_run = {
        "protocol",
        "protocol_manifest_sha256",
        "protocol_manifest",
        "development_history",
        "config",
        "environment",
        "selected_scenarios",
        "selected_data_seeds",
        "is_complete_canonical_grid",
        "records",
        "cell_engineering_status",
        "runtime_seconds",
    }
    if not isinstance(run, Mapping):
        raise TypeError("run must be a mapping.")
    missing = sorted(required_run.difference(run))
    if missing:
        raise ValueError(f"Probability run is missing keys: {missing}.")
    records = run["records"]
    if not isinstance(records, Sequence) or not records:
        raise ValueError("Probability run must contain at least one fitted record.")
    required_record = {
        "scenario",
        "data_seed",
        "initialization_replicate",
        "is_initialization_sensitivity",
        "factors",
        "cardinalities",
        "schema",
        "anchor_outcome",
        "focal_class_index",
        "focal_training_count",
        "focal_test_count",
        "model_diagnostics",
        "focal_true_probability_bands",
        "oracle_metrics",
        "quadrature_convergence",
        "probability_validity",
        "engineering_checks",
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping.")
        absent = sorted(required_record.difference(record))
        if absent:
            raise ValueError(f"records[{index}] is missing keys: {absent}.")
        models = record["model_diagnostics"].get("models", {})
        oracle_metrics = record["oracle_metrics"]
        for model in CANDIDATE_MODELS:
            if model not in models or model not in oracle_metrics:
                raise ValueError(
                    f"records[{index}] lacks required candidate model {model!r}."
                )


def _base_records(run: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        record
        for record in run["records"]
        if int(record["initialization_replicate"]) == 0
    ]


def _model_diagnostic_summary(record: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    return record["model_diagnostics"]["models"][model]["summary"]


def _oracle_summary(record: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    return record["oracle_metrics"][model]["summary"]


def _oracle_focal(record: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    return record["oracle_metrics"][model]["focal"]


def _focal_diagnostic(record: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    outcome = str(record["anchor_outcome"])
    class_index = int(record["focal_class_index"])
    return next(
        item
        for item in record["model_diagnostics"]["models"][model]["classes"]
        if str(item["outcome"]) == outcome and int(item["class_index"]) == class_index
    )


def _diagnostic_metric(key: str) -> MetricGetter:
    def getter(record: Mapping[str, Any], model: str) -> Optional[float]:
        summary = _model_diagnostic_summary(record, model)
        value = summary.get(key)
        if value is None and key.startswith("macro_"):
            value = summary.get(f"partial_{key}")
        return None if value is None else float(value)

    return getter


def _oracle_summary_metric(key: str) -> MetricGetter:
    return lambda record, model: (
        None
        if _oracle_summary(record, model).get(key) is None
        else float(_oracle_summary(record, model)[key])
    )


def _oracle_focal_metric(key: str) -> MetricGetter:
    return lambda record, model: (
        None
        if _oracle_focal(record, model).get(key) is None
        else float(_oracle_focal(record, model)[key])
    )


def _focal_diagnostic_metric(key: str) -> MetricGetter:
    return lambda record, model: (
        None
        if _focal_diagnostic(record, model).get(key) is None
        else float(_focal_diagnostic(record, model)[key])
    )


def _unique_sorted(values: Sequence[Any]) -> List[Any]:
    unique = list(dict.fromkeys(values))
    try:
        return sorted(unique, key=float)
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def _group_values(
    records: Sequence[Mapping[str, Any]],
    model: str,
    getter: MetricGetter,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> Tuple[float, float]:
    values = [getter(record, model) for record in records if predicate(record)]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return math.nan, 0.0
    return _mean_95_half_width(numeric)


def _set_fixed_log_sample_ticks(axis: Any, sample_sizes: Sequence[int]) -> None:
    """Show only the sample sizes that were actually evaluated."""

    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(sample_sizes))
    axis.xaxis.set_major_formatter(
        FixedFormatter([f"{int(value):,}" for value in sample_sizes])
    )
    axis.xaxis.set_minor_locator(NullLocator())


def _plot_core_metric_group(
    records: Sequence[Mapping[str, Any]],
    specifications: Sequence[Tuple[str, MetricGetter]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> Optional[Path]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    if not core:
        return None
    prevalences = _unique_sorted(
        [float(record["factors"]["focal_prevalence"]) for record in core]
    )
    sample_sizes = _unique_sorted(
        [int(record["factors"]["n_train"]) for record in core]
    )
    cardinalities = _unique_sorted(
        [int(record["factors"]["maximum_cardinality"]) for record in core]
    )
    plt = _pyplot()
    rows = len(specifications) * len(CANDIDATE_MODELS)
    fig, axes = plt.subplots(
        rows,
        len(prevalences),
        figsize=(4.25 * len(prevalences), 2.75 * rows),
        squeeze=False,
        sharex=True,
    )
    cmap = plt.get_cmap("viridis")
    for metric_index, (metric_label, getter) in enumerate(specifications):
        for model_index, model in enumerate(CANDIDATE_MODELS):
            row_index = metric_index * len(CANDIDATE_MODELS) + model_index
            for prevalence_index, prevalence in enumerate(prevalences):
                axis = axes[row_index, prevalence_index]
                for k_index, cardinality in enumerate(cardinalities):
                    means = []
                    lower_errors = []
                    upper_errors = []
                    for sample_size in sample_sizes:
                        mean, half_width = _group_values(
                            core,
                            model,
                            getter,
                            lambda record, n=sample_size, q=prevalence, k=cardinality: (
                                int(record["factors"]["n_train"]) == n
                                and float(record["factors"]["focal_prevalence"]) == q
                                and int(record["factors"]["maximum_cardinality"]) == k
                            ),
                        )
                        means.append(mean)
                        # Every metric in these core panels is nonnegative. A
                        # symmetric small-sample t interval can cross zero, so
                        # truncate only its displayed lower whisker at the
                        # metric's natural boundary.
                        lower_errors.append(min(half_width, max(mean, 0.0)))
                        upper_errors.append(half_width)
                    axis.errorbar(
                        sample_sizes,
                        means,
                        yerr=np.vstack((lower_errors, upper_errors)),
                        marker="o",
                        linewidth=1.3,
                        capsize=2,
                        color=cmap(k_index / max(1, len(cardinalities) - 1)),
                        label=f"K={cardinality}",
                    )
                _set_fixed_log_sample_ticks(axis, sample_sizes)
                axis.grid(alpha=0.18)
                if row_index == 0:
                    axis.set_title(f"focal prevalence={prevalence:g}")
                if prevalence_index == 0:
                    axis.set_ylabel(
                        f"{metric_label}\n{MODEL_LABELS.get(model, model)}",
                        fontsize=9,
                    )
                if row_index == rows - 1:
                    axis.set_xlabel("training sample size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=len(labels),
        )
    fig.suptitle(
        title + "\nmeans with 95% t intervals across fixed data seeds",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.875, hspace=0.32, wspace=0.20)
    return _save_figure(fig, output_path, dpi)


def _plot_heterogeneity(
    records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    selected = [
        record for record in records if record["factors"]["design"] == "heterogeneity"
    ]
    if not selected:
        return None
    sample_sizes = _unique_sorted(
        [int(record["factors"]["n_train"]) for record in selected]
    )
    specifications: Tuple[Tuple[str, MetricGetter], ...] = (
        ("focal probability RMSE", _oracle_focal_metric("rmse")),
        ("mean outcome total variation", _oracle_summary_metric("mean_outcome_total_variation")),
        ("mean outcome oracle KL", _oracle_summary_metric("mean_outcome_kl_regret")),
        (
            "expected Brier regret",
            _oracle_summary_metric("mean_outcome_expected_brier_regret"),
        ),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), squeeze=False)
    for axis, (label, getter) in zip(axes.flat, specifications):
        for model in CANDIDATE_MODELS:
            means = []
            errors = []
            for sample_size in sample_sizes:
                homogeneous = {
                    int(record["data_seed"]): record
                    for record in selected
                    if int(record["factors"]["n_train"]) == sample_size
                    and str(record["factors"]["variant"]) == "homogeneous"
                }
                heterogeneous = {
                    int(record["data_seed"]): record
                    for record in selected
                    if int(record["factors"]["n_train"]) == sample_size
                    and str(record["factors"]["variant"]) == "heterogeneous"
                }
                common_seeds = sorted(set(homogeneous).intersection(heterogeneous))
                differences = [
                    float(getter(heterogeneous[seed], model))
                    - float(getter(homogeneous[seed], model))
                    for seed in common_seeds
                ]
                if differences:
                    mean, half_width = _mean_95_half_width(differences)
                    jitter = 0.97 if model == "cvae" else 1.03
                    axis.scatter(
                        np.repeat(sample_size * jitter, len(differences)),
                        differences,
                        s=15,
                        alpha=0.35,
                        color=MODEL_COLORS[model],
                    )
                else:
                    mean, half_width = math.nan, 0.0
                means.append(mean)
                errors.append(half_width)
            axis.errorbar(
                sample_sizes,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        _set_fixed_log_sample_ticks(axis, sample_sizes)
        axis.set_title(f"heterogeneous - homogeneous:\n{label}", fontsize=10)
        axis.set_xlabel("training sample size")
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        fontsize=9,
    )
    fig.suptitle(
        "Decoder-width-matched schemas (same J and sum K; not entropy matched)\n"
        "paired seed differences, means with 95% t intervals; faint points are seeds",
        fontsize=13,
        y=0.965,
    )
    fig.subplots_adjust(top=0.77, hspace=0.40, wspace=0.25)
    return _save_figure(fig, output_path, dpi)


def _plot_rho_control(
    records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    controls = [
        record for record in records if record["factors"]["design"] == "rho_control"
    ]
    if not controls:
        return None
    reference = controls[0]["factors"]
    selected = [
        record
        for record in records
        if record["factors"]["design"] in ("core", "rho_control")
        and int(record["factors"]["n_train"]) == int(reference["n_train"])
        and float(record["factors"]["focal_prevalence"])
        == float(reference["focal_prevalence"])
        and int(record["factors"]["maximum_cardinality"])
        == int(reference["maximum_cardinality"])
    ]
    rhos = _unique_sorted([float(record["factors"]["copula_rho"]) for record in selected])
    if len(rhos) < 2:
        return None
    specifications: Tuple[Tuple[str, MetricGetter], ...] = (
        ("focal probability RMSE", _oracle_focal_metric("rmse")),
        ("mean outcome total variation", _oracle_summary_metric("mean_outcome_total_variation")),
        ("mean outcome oracle KL", _oracle_summary_metric("mean_outcome_kl_regret")),
        ("mean classwise ECE", _diagnostic_metric("mean_classwise_ece")),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), squeeze=False)
    for axis, (label, getter) in zip(axes.flat, specifications):
        for model in CANDIDATE_MODELS:
            means = []
            errors = []
            for rho in rhos:
                mean, standard_error = _group_values(
                    selected,
                    model,
                    getter,
                    lambda record, value=rho: float(
                        record["factors"]["copula_rho"]
                    )
                    == value,
                )
                means.append(mean)
                errors.append(standard_error)
            axis.errorbar(
                rhos,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_xticks(rhos)
        axis.set_title(label)
        axis.set_xlabel("Gaussian-copula correlation")
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(labels),
    )
    fig.suptitle(
        "Marginal probability recovery with and without residual dependence\n"
        "means with 95% t intervals across fixed data seeds",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.25)
    return _save_figure(fig, output_path, dpi)


def _plot_quadrature_diagnostics(
    records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    if not core:
        core = list(records)
    if not core:
        return None
    specifications = (
        ("mean_absolute_probability_change", "mean absolute change"),
        ("rmse_probability_change", "RMSE change"),
        ("p99_absolute_probability_change", "p99 absolute change"),
        ("maximum_absolute_probability_change", "maximum absolute change"),
    )
    sample_sizes = _unique_sorted(
        [int(record["factors"]["n_train"]) for record in core]
    )
    cardinalities = _unique_sorted(
        [int(record["factors"]["maximum_cardinality"]) for record in core]
    )
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for axis, (key, label) in zip(axes.flat, specifications):
        for k_index, cardinality in enumerate(cardinalities):
            means = []
            errors = []
            for sample_size in sample_sizes:
                values = [
                    float(record["quadrature_convergence"][key])
                    for record in core
                    if int(record["factors"]["n_train"]) == sample_size
                    and int(record["factors"]["maximum_cardinality"]) == cardinality
                ]
                if values:
                    mean = float(np.mean(values))
                    error = (
                        mean - float(np.min(values)),
                        float(np.max(values)) - mean,
                    )
                else:
                    mean, error = math.nan, (0.0, 0.0)
                means.append(mean)
                errors.append(error)
            asymmetric_error = np.asarray(errors, dtype=np.float64).T
            axis.errorbar(
                sample_sizes,
                means,
                yerr=asymmetric_error,
                marker="o",
                capsize=2,
                linewidth=1.3,
                color=cmap(k_index / max(1, len(cardinalities) - 1)),
                label=f"K={cardinality}",
            )
        thresholds = []
        check_name = {
            "mean_absolute_probability_change": "quadrature_mean_absolute_change",
            "p99_absolute_probability_change": "quadrature_p99_absolute_change",
        }.get(key)
        if check_name is not None:
            thresholds = [
                float(record["engineering_checks"]["checks"][check_name]["threshold"])
                for record in core
            ]
        if thresholds and np.allclose(thresholds, thresholds[0]):
            axis.axhline(
                thresholds[0], color="#a83232", linestyle="--", linewidth=1.0,
                label="engineering tolerance",
            )
        _set_fixed_log_sample_ticks(axis, sample_sizes)
        axis.set_yscale("log")
        axis.set_title(label)
        axis.set_xlabel("training sample size")
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(4, len(labels)),
        )
    fig.suptitle(
        "Numerical sensitivity of fitted marginal probabilities: GH21 versus GH31\n"
        "means and full ranges across focal prevalences and fixed seeds",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.87, hspace=0.34, wspace=0.25)
    return _save_figure(fig, output_path, dpi)


def _plot_focal_true_probability_bands(
    records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    if not core:
        return None
    prevalences = _unique_sorted(
        [float(record["factors"]["focal_prevalence"]) for record in core]
    )
    available = [
        band
        for record in core
        for model in CANDIDATE_MODELS
        for band in record.get("focal_true_probability_bands", {}).get(model, [])
    ]
    if not available:
        return None
    labels = [
        item["label"]
        for item in sorted(
            {item["label"]: item for item in available}.values(),
            key=lambda item: float(item["lower"]),
        )
    ]
    specifications = (
        ("bias", "signed error (p-hat - p-true)"),
        ("mae", "mean absolute error"),
        ("rmse", "root mean squared error"),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(specifications),
        len(prevalences),
        figsize=(4.5 * len(prevalences), 3.1 * len(specifications)),
        squeeze=False,
        sharex=True,
    )
    x = np.arange(len(labels))
    for metric_index, (key, metric_label) in enumerate(specifications):
        for prevalence_index, prevalence in enumerate(prevalences):
            axis = axes[metric_index, prevalence_index]
            for model in CANDIDATE_MODELS:
                means = []
                lower_errors = []
                upper_errors = []
                for label in labels:
                    values = [
                        float(band[key])
                        for record in core
                        if float(record["factors"]["focal_prevalence"]) == prevalence
                        for band in record["focal_true_probability_bands"][model]
                        if band["label"] == label
                    ]
                    if values:
                        mean = float(np.mean(values))
                        means.append(mean)
                        lower_errors.append(mean - float(np.min(values)))
                        upper_errors.append(float(np.max(values)) - mean)
                    else:
                        means.append(math.nan)
                        lower_errors.append(0.0)
                        upper_errors.append(0.0)
                axis.errorbar(
                    x,
                    means,
                    yerr=np.vstack((lower_errors, upper_errors)),
                    marker="o",
                    capsize=2,
                    linewidth=1.35,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
            if key == "bias":
                axis.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
            if metric_index == 0:
                axis.set_title(f"target prevalence={prevalence:g}")
            if prevalence_index == 0:
                axis.set_ylabel(metric_label)
            if metric_index == len(specifications) - 1:
                axis.set_xticks(x)
                axis.set_xticklabels(labels, rotation=35, ha="right")
                axis.set_xlabel("known p-true range")
            axis.grid(axis="y", alpha=0.18)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(legend_labels),
    )
    fig.suptitle(
        "Focal-class error by known conditional-probability band\n"
        "means and full fit-level ranges across n, K, and fixed seeds",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.23, bottom=0.13)
    return _save_figure(fig, output_path, dpi)


def _plot_low_prevalence_training_counts(
    records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    core = [
        record
        for record in records
        if record["factors"]["design"] == "core"
        and float(record["factors"]["focal_prevalence"]) <= 0.05
    ]
    if not core:
        return None
    prevalences = _unique_sorted(
        [float(record["factors"]["focal_prevalence"]) for record in core]
    )
    cardinalities = _unique_sorted(
        [int(record["factors"]["maximum_cardinality"]) for record in core]
    )
    sample_sizes = _unique_sorted(
        [int(record["factors"]["n_train"]) for record in core]
    )
    markers = ("o", "s", "^")
    marker_by_n = {
        sample_size: markers[index % len(markers)]
        for index, sample_size in enumerate(sample_sizes)
    }
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(prevalences), 2, figsize=(10.5, 3.7 * len(prevalences)),
        squeeze=False,
    )
    cmap = plt.get_cmap("viridis")
    color_by_k = {
        cardinality: cmap(index / max(1, len(cardinalities) - 1))
        for index, cardinality in enumerate(cardinalities)
    }
    for prevalence_index, prevalence in enumerate(prevalences):
        subset = [
            record
            for record in core
            if float(record["factors"]["focal_prevalence"]) == prevalence
        ]
        for metric_index, (key, label) in enumerate(
            (("mae", "focal probability MAE"), ("bias", "focal probability bias"))
        ):
            axis = axes[prevalence_index, metric_index]
            for record in subset:
                cardinality = int(record["factors"]["maximum_cardinality"])
                sample_size = int(record["factors"]["n_train"])
                axis.scatter(
                    int(record["focal_training_count"]),
                    float(_oracle_focal(record, "cvae")[key]),
                    marker=marker_by_n[sample_size],
                    color=color_by_k[cardinality],
                    s=33,
                    alpha=0.68,
                )
            if key == "bias":
                axis.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
            axis.set_title(f"target prevalence={prevalence:g}: {label}")
            axis.set_xlabel("observed focal events in training data")
            axis.set_ylabel(label)
            axis.grid(alpha=0.18)
    from matplotlib.lines import Line2D

    color_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=color_by_k[k],
            label=f"K={k}",
        )
        for k in cardinalities
    ]
    marker_handles = [
        Line2D(
            [0], [0], marker=marker_by_n[n], linestyle="none", color="0.35",
            label=f"n={n}",
        )
        for n in sample_sizes
    ]
    fig.legend(
        color_handles + marker_handles,
        [handle.get_label() for handle in color_handles + marker_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(color_handles + marker_handles),
        fontsize=8,
    )
    fig.suptitle(
        "Low-prevalence CVAE error versus the focal event count actually seen in training",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.88, hspace=0.40, wspace=0.27)
    return _save_figure(fig, output_path, dpi)


def _plot_initialization_diagnostics(
    all_records: Sequence[Mapping[str, Any]], output_path: Path, *, dpi: int
) -> Optional[Path]:
    sensitivity_scenarios = {
        str(record["scenario"])
        for record in all_records
        if int(record["initialization_replicate"]) > 0
    }
    if not sensitivity_scenarios:
        return None
    sensitivity = [
        record
        for record in all_records
        if str(record["scenario"]) in sensitivity_scenarios
        and int(record["data_seed"])
        == min(
            int(candidate["data_seed"])
            for candidate in all_records
            if str(candidate["scenario"]) == str(record["scenario"])
            and int(candidate["initialization_replicate"]) > 0
        )
    ]
    specifications: Tuple[Tuple[str, MetricGetter], ...] = (
        ("focal probability RMSE", _oracle_focal_metric("rmse")),
        ("focal p95 absolute error", _oracle_focal_metric("absolute_error_p95")),
        ("mean outcome total variation", _oracle_summary_metric("mean_outcome_total_variation")),
        ("mean outcome oracle KL", _oracle_summary_metric("mean_outcome_kl_regret")),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for axis, (label, getter) in zip(axes.flat, specifications):
        for scenario_index, scenario in enumerate(sorted(sensitivity_scenarios)):
            scenario_records = sorted(
                [record for record in sensitivity if str(record["scenario"]) == scenario],
                key=lambda record: int(record["initialization_replicate"]),
            )
            axis.plot(
                [int(record["initialization_replicate"]) for record in scenario_records],
                [getter(record, "cvae") for record in scenario_records],
                marker="o",
                linewidth=1.3,
                color=cmap(scenario_index % 10),
                label=scenario,
            )
        axis.set_title(label)
        axis.set_xlabel("initialization/training replicate")
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=1,
        fontsize=8,
    )
    fig.suptitle(
        "CVAE sensitivity to initialization on the same fixed data split",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.82, hspace=0.34, wspace=0.25)
    return _save_figure(fig, output_path, dpi)


def _plot_sentinel_probability_agreement(
    record: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    """Plot individual focal-class p-hat against known p-true."""

    payload = record["plot_payload"]
    anchor = str(record["anchor_outcome"])
    class_index = int(record["focal_class_index"])
    actual = np.asarray(
        payload["oracle_probabilities"][anchor], dtype=np.float64
    )[:, class_index]
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(CANDIDATE_MODELS), 2, figsize=(9.4, 4.65 * len(CANDIDATE_MODELS)),
        squeeze=False,
    )
    for model_index, model in enumerate(CANDIDATE_MODELS):
        predicted = np.asarray(
            payload["model_probabilities"][model][anchor], dtype=np.float64
        )[:, class_index]
        for column_index, limit in enumerate((1.0, 0.20)):
            axis = axes[model_index, column_index]
            selected = actual <= limit
            x = actual[selected]
            y = predicted[selected]
            within = y <= limit
            if int(within.sum()) >= 100:
                axis.hexbin(
                    x[within],
                    y[within],
                    gridsize=35,
                    extent=(0.0, limit, 0.0, limit),
                    mincnt=1,
                    cmap="Blues",
                )
            else:
                axis.scatter(
                    x[within], y[within], s=10, alpha=0.45,
                    color=MODEL_COLORS[model], rasterized=True,
                )
            overflow = int((~within).sum())
            if overflow:
                axis.scatter(
                    x[~within],
                    np.repeat(limit, overflow),
                    marker="^",
                    s=13,
                    alpha=0.7,
                    color="#a83232",
                    rasterized=True,
                )
            axis.plot(
                [0.0, limit], [0.0, limit], linestyle="--", color="0.4",
                linewidth=1.0,
            )
            axis.set_xlim(0.0, limit)
            axis.set_ylim(0.0, limit)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.14)
            error = y - x
            error_text = (
                f"n={error.size}, MAE={np.abs(error).mean():.4f}, "
                f"RMSE={np.sqrt(np.square(error).mean()):.4f}, overflow={overflow}"
                if error.size
                else "n=0 in displayed p-true range"
            )
            axis.set_title(
                f"{MODEL_LABELS[model]}"
                + ("; full range" if limit == 1.0 else f"; p-true <= {limit:g}")
                + f"\n{error_text}",
                fontsize=9,
            )
            axis.set_xlabel("known DGP conditional probability (p-true)")
            axis.set_ylabel("fitted conditional probability (p-hat)")
    factors = record["factors"]
    fig.suptitle(
        f"Individual focal probabilities: {record['scenario']}\n"
        f"n={factors['n_train']}, target prevalence={factors['focal_prevalence']}, "
        f"cardinalities={record['cardinalities']}",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(top=0.89, hspace=0.52, wspace=0.34)
    return _save_figure(fig, output_path, dpi)


def _sentinel_calibration_figures(
    record: Mapping[str, Any], output_dir: Path, *, dpi: int
) -> List[Path]:
    payload = record["plot_payload"]
    full_test = payload.get("anchor_reliability_bins_full_test")
    if not isinstance(full_test, Mapping):
        raise ValueError(
            f"Sentinel {record['scenario']!r} lacks full-test anchor reliability bins."
        )
    subset_comparison = {
        "models": {
            model: {"classes": list(full_test[model])}
            for model in CANDIDATE_MODELS
        },
        "has_oracle": True,
    }
    paths = plot_comparative_calibration_pages(
        subset_comparison,
        output_dir,
        filename_prefix=f"{_slug(record['scenario'])}_anchor",
        maximum_panels_per_page=10,
        dpi=dpi,
    )
    paths.extend(
        plot_comparative_calibration_pages(
            subset_comparison,
            output_dir,
            filename_prefix=f"{_slug(record['scenario'])}_anchor",
            maximum_panels_per_page=10,
            probability_limit=0.20,
            dpi=dpi,
        )
    )
    return paths


def generate_probability_report_figures(
    run: Mapping[str, Any], figure_dir: Path, *, dpi: int = 150
) -> List[Dict[str, Any]]:
    """Generate every report figure and return ordered Markdown metadata."""

    validate_probability_run(run)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = _base_records(run)
    figures: List[Dict[str, Any]] = []

    core_groups = (
        (
            "core_discrimination",
            "Discrimination across sample size, prevalence, and cardinality",
            (
                ("defined-class macro ROC-AUC", _diagnostic_metric("macro_one_vs_rest_auc")),
                ("defined-class macro average precision", _diagnostic_metric("macro_one_vs_rest_pr_auc")),
            ),
        ),
        (
            "core_focal_discrimination",
            "Focal-class discrimination (AP null is observed focal prevalence)",
            (
                ("focal one-vs-rest ROC-AUC", _focal_diagnostic_metric("auc")),
                ("focal average precision", _focal_diagnostic_metric("pr_auc")),
            ),
        ),
        (
            "core_realized_scores",
            "Held-out proper scores and empirical calibration",
            (
                ("mean outcome NLL", _diagnostic_metric("mean_outcome_nll")),
                ("mean outcome multiclass Brier", _diagnostic_metric("mean_outcome_multiclass_brier")),
                ("mean classwise ECE", _diagnostic_metric("mean_classwise_ece")),
            ),
        ),
        (
            "core_oracle_regret",
            "Direct error against known conditional probabilities",
            (
                ("mean outcome total variation", _oracle_summary_metric("mean_outcome_total_variation")),
                ("mean outcome KL/log-score regret", _oracle_summary_metric("mean_outcome_kl_regret")),
                ("mean outcome expected Brier regret", _oracle_summary_metric("mean_outcome_expected_brier_regret")),
            ),
        ),
        (
            "core_focal_error",
            "Focal-class probability error across the core grid",
            (
                ("focal MAE", _oracle_focal_metric("mae")),
                ("focal RMSE", _oracle_focal_metric("rmse")),
                ("focal p95 absolute error", _oracle_focal_metric("absolute_error_p95")),
            ),
        ),
    )
    for filename, title, specifications in core_groups:
        path = _plot_core_metric_group(
            base,
            specifications,
            figure_dir / f"{filename}.png",
            title=title,
            dpi=dpi,
        )
        if path is not None:
            figures.append({"section": "Core factorial results", "path": path, "caption": title})

    band_path = _plot_focal_true_probability_bands(
        base, figure_dir / "focal_true_probability_bands.png", dpi=dpi
    )
    if band_path is not None:
        figures.append(
            {
                "section": "Core factorial results",
                "path": band_path,
                "caption": "Focal-class error by known p-true range",
            }
        )
    count_path = _plot_low_prevalence_training_counts(
        base, figure_dir / "low_prevalence_training_counts.png", dpi=dpi
    )
    if count_path is not None:
        figures.append(
            {
                "section": "Core factorial results",
                "path": count_path,
                "caption": "Low-prevalence error versus realized focal training events",
            }
        )

    optional_plots = (
        (
            _plot_heterogeneity,
            figure_dir / "heterogeneity_comparison.png",
            "Decoder-width-matched cardinality heterogeneity comparison",
            "Cardinality heterogeneity",
        ),
        (
            _plot_rho_control,
            figure_dir / "rho_control.png",
            "Residual-dependence control",
            "Dependence control",
        ),
        (
            _plot_quadrature_diagnostics,
            figure_dir / "quadrature_diagnostics.png",
            "Gauss-Hermite integration diagnostics",
            "Numerical diagnostics",
        ),
    )
    for function, path, caption, section in optional_plots:
        generated = function(base, path, dpi=dpi)
        if generated is not None:
            figures.append({"section": section, "path": generated, "caption": caption})
    initialization = _plot_initialization_diagnostics(
        run["records"], figure_dir / "initialization_diagnostics.png", dpi=dpi
    )
    if initialization is not None:
        figures.append(
            {
                "section": "Initialization sensitivity",
                "path": initialization,
                "caption": "Initialization/training replicate sensitivity",
            }
        )

    for record in base:
        if "plot_payload" not in record:
            continue
        agreement = _plot_sentinel_probability_agreement(
            record,
            figure_dir / f"{_slug(record['scenario'])}_probability_agreement.png",
            dpi=dpi,
        )
        figures.append(
            {
                "section": "Sentinel probability and calibration plots",
                "path": agreement,
                "caption": f"{record['scenario']}: individual p-hat versus p-true",
            }
        )
        for page_index, path in enumerate(
            _sentinel_calibration_figures(record, figure_dir, dpi=dpi), start=1
        ):
            is_low_probability = "low_probability" in path.name
            figures.append(
                {
                    "section": "Sentinel probability and calibration plots",
                    "path": path.resolve(),
                    "caption": (
                        f"{record['scenario']}: observed and oracle calibration "
                        f"for every anchor level"
                        + (" (0-0.20 zoom)" if is_low_probability else "")
                        + f" (page {page_index})"
                    ),
                }
            )
    return figures


def _range_summary(values: Sequence[float], digits: int = 4) -> str:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return "NA"
    return (
        f"{np.median(array):.{digits}f} "
        f"[{np.min(array):.{digits}f}, {np.max(array):.{digits}f}]"
    )


def _ranking_coverage(
    records: Sequence[Mapping[str, Any]], model: str, key: str
) -> Tuple[int, int]:
    defined = 0
    required = 0
    for record in records:
        classes = record["model_diagnostics"]["models"][model]["classes"]
        required += len(classes)
        defined += sum(item.get(key) is not None for item in classes)
    return defined, required


def _model_descriptive_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| model | focal MAE | focal RMSE | focal p95 absolute error | mean-outcome TV | oracle KL regret | expected Brier regret | held-out multiclass Brier | defined-class macro ROC-AUC | ROC coverage | defined-class macro AP | AP coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in CANDIDATE_MODELS:
        focal_mae = [float(_oracle_focal(record, model)["mae"]) for record in records]
        focal_rmse = [float(_oracle_focal(record, model)["rmse"]) for record in records]
        focal_p95 = [
            float(_oracle_focal(record, model)["absolute_error_p95"])
            for record in records
        ]
        tv = [
            float(_oracle_summary(record, model)["mean_outcome_total_variation"])
            for record in records
        ]
        kl = [
            float(_oracle_summary(record, model)["mean_outcome_kl_regret"])
            for record in records
        ]
        brier_regret = [
            float(_oracle_summary(record, model)["mean_outcome_expected_brier_regret"])
            for record in records
        ]
        brier = [
            float(_model_diagnostic_summary(record, model)["mean_outcome_multiclass_brier"])
            for record in records
        ]
        auc_getter = _diagnostic_metric("macro_one_vs_rest_auc")
        ap_getter = _diagnostic_metric("macro_one_vs_rest_pr_auc")
        auc_values = [
            value for record in records if (value := auc_getter(record, model)) is not None
        ]
        ap_values = [
            value for record in records if (value := ap_getter(record, model)) is not None
        ]
        auc_defined, auc_required = _ranking_coverage(records, model, "auc")
        ap_defined, ap_required = _ranking_coverage(records, model, "pr_auc")
        lines.append(
            f"| {MODEL_LABELS[model]} | {_range_summary(focal_mae)} | "
            f"{_range_summary(focal_rmse)} | {_range_summary(focal_p95)} | "
            f"{_range_summary(tv)} | {_range_summary(kl)} | "
            f"{_range_summary(brier_regret)} | {_range_summary(brier)} | "
            f"{_range_summary(auc_values)} | {auc_defined}/{auc_required} | "
            f"{_range_summary(ap_values)} | {ap_defined}/{ap_required} |"
        )
    return lines


def _plain_english_summary(records: Sequence[Mapping[str, Any]]) -> str:
    cvae_focal = np.asarray(
        [float(_oracle_focal(record, "cvae")["mae"]) for record in records]
    )
    baseline_focal = np.asarray(
        [float(_oracle_focal(record, "independent_softmax")["mae"]) for record in records]
    )
    cvae_tv = np.asarray(
        [
            float(_oracle_summary(record, "cvae")["mean_outcome_total_variation"])
            for record in records
        ]
    )
    baseline_tv = np.asarray(
        [
            float(
                _oracle_summary(record, "independent_softmax")[
                    "mean_outcome_total_variation"
                ]
            )
            for record in records
        ]
    )
    focal_wins = int(np.sum(cvae_focal < baseline_focal))
    tv_wins = int(np.sum(cvae_tv < baseline_tv))
    worst_index = int(np.argmax(cvae_focal))
    worst = records[worst_index]
    return (
        f"Across the {len(records)} base fits shown here, the CVAE's median focal-class "
        f"absolute probability error was {np.median(cvae_focal):.4f}, compared with "
        f"{np.median(baseline_focal):.4f} for independent softmax. The CVAE had lower "
        f"focal MAE in {focal_wins}/{len(records)} matched fits and lower equal-outcome "
        f"total variation in {tv_wins}/{len(records)}. Its worst fit-level focal MAE "
        f"was {cvae_focal[worst_index]:.4f} in `{worst['scenario']}` "
        f"(seed {worst['data_seed']}). These are descriptive comparisons over the "
        "declared regimes, not a claim of validity outside them."
    )


def _focal_score_by_prevalence_table(
    records: Sequence[Mapping[str, Any]],
) -> List[str]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    lines = [
        "| target prevalence | CVAE focal Brier | oracle focal Brier | CVAE focal ECE | CVAE focal ROC-AUC | CVAE focal AP | observed prevalence (AP null) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prevalence in _unique_sorted(
        [float(record["factors"]["focal_prevalence"]) for record in core]
    ):
        group = [
            record
            for record in core
            if float(record["factors"]["focal_prevalence"]) == prevalence
        ]

        def median_diagnostic(model: str, key: str) -> Optional[float]:
            values = []
            for record in group:
                if model not in record["model_diagnostics"]["models"]:
                    continue
                value = _focal_diagnostic(record, model).get(key)
                if value is not None:
                    values.append(float(value))
            return float(np.median(values)) if values else None

        observed = np.median(
            [
                int(record["focal_test_count"]) / int(record["factors"]["n_test"])
                for record in group
            ]
        )
        lines.append(
            f"| {prevalence:g} | {_format(median_diagnostic('cvae', 'binary_brier'), 5)} | "
            f"{_format(median_diagnostic('oracle', 'binary_brier'), 5)} | "
            f"{_format(median_diagnostic('cvae', 'ece'))} | "
            f"{_format(median_diagnostic('cvae', 'auc'), 3)} | "
            f"{_format(median_diagnostic('cvae', 'pr_auc'), 3)} | {observed:.3f} |"
        )
    return lines


def _core_cardinality_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    lines = [
        "| n | K | CVAE focal MAE | CVAE mean-outcome TV | softmax mean-outcome TV | CVAE KL regret |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    sample_sizes = _unique_sorted(
        [int(record["factors"]["n_train"]) for record in core]
    )
    cardinalities = _unique_sorted(
        [int(record["factors"]["maximum_cardinality"]) for record in core]
    )
    for sample_size in sample_sizes:
        for cardinality in cardinalities:
            group = [
                record
                for record in core
                if int(record["factors"]["n_train"]) == sample_size
                and int(record["factors"]["maximum_cardinality"]) == cardinality
            ]
            if not group:
                continue
            median = lambda values: float(np.median(list(values)))
            lines.append(
                f"| {sample_size:,} | {cardinality} | "
                f"{median(_oracle_focal(record, 'cvae')['mae'] for record in group):.4f} | "
                f"{median(_oracle_summary(record, 'cvae')['mean_outcome_total_variation'] for record in group):.4f} | "
                f"{median(_oracle_summary(record, 'independent_softmax')['mean_outcome_total_variation'] for record in group):.4f} | "
                f"{median(_oracle_summary(record, 'cvae')['mean_outcome_kl_regret'] for record in group):.4f} |"
            )
    return lines


def _factor_finding_lines(records: Sequence[Mapping[str, Any]]) -> List[str]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    if not core:
        return []
    lines = ["## What the experiment says about the requested circumstances", ""]

    lines.extend(["### Baseline probability and calibration", ""])
    for prevalence in _unique_sorted(
        [float(record["factors"]["focal_prevalence"]) for record in core]
    ):
        group = [
            record
            for record in core
            if float(record["factors"]["focal_prevalence"]) == prevalence
        ]
        focal_mae = float(
            np.median([_oracle_focal(record, "cvae")["mae"] for record in group])
        )
        focal_p95 = float(
            np.median(
                [
                    _oracle_focal(record, "cvae")["absolute_error_p95"]
                    for record in group
                ]
            )
        )
        absolute_bias = float(
            np.median(
                [abs(_oracle_focal(record, "cvae")["bias"]) for record in group]
            )
        )
        lines.append(
            f"- With a {100 * prevalence:g}% population-average focal probability, "
            f"the median individual-probability MAE was {focal_mae:.4f} "
            f"({100 * focal_mae:.2f} percentage points), the median 95th-percentile "
            f"absolute error was {focal_p95:.4f}, and the median absolute fit-level "
            f"bias was {absolute_bias:.4f}."
        )
    lines.extend(
        [
            "",
            (
                "A smaller absolute error at 1% does not mean rare outcomes were easy: "
                "an MAE near 0.0056 is about half of a 1% population-average risk. "
                "The target prevalence is a population mean; each person's p-true varies with X."
            ),
            "",
            *_focal_score_by_prevalence_table(core),
            "",
            (
                "The raw focal Brier score stays numerically close to the oracle because "
                "irreducible outcome noise and prevalence dominate that score. Direct "
                "p-hat-versus-p-true errors and Brier regret are more sensitive to the "
                "conditional-probability error in this simulation. AUC and AP can look "
                "respectable while the probabilities themselves are miscalibrated."
            ),
            "",
        ]
    )

    band_targets = {
        0.01: "[0.05, 0.1)",
        0.05: "[0.2, 0.5)",
        0.20: "[0.5, 1]",
    }
    shrinkage_parts = []
    for prevalence, label in band_targets.items():
        values = [
            band
            for record in core
            if math.isclose(
                float(record["factors"]["focal_prevalence"]), prevalence
            )
            for band in record["focal_true_probability_bands"]["cvae"]
            if str(band["label"]) == label
        ]
        if values:
            shrinkage_parts.append(
                f"q={prevalence:g}, p-true {label}: MAE "
                f"{np.mean([float(value['mae']) for value in values]):.4f}, bias "
                f"{np.mean([float(value['bias']) for value in values]):+.4f}"
            )
    if shrinkage_parts:
        lines.extend(
            [
                (
                    "The main calibration pattern is compression toward the population "
                    "mean: high individualized probabilities were underpredicted. "
                    + "; ".join(shrinkage_parts)
                    + "."
                ),
                "",
            ]
        )

    lines.extend(["### Training sample size", ""])
    n_parts = []
    baseline_parts = []
    for sample_size in _unique_sorted(
        [int(record["factors"]["n_train"]) for record in core]
    ):
        group = [
            record
            for record in core
            if int(record["factors"]["n_train"]) == sample_size
        ]
        n_parts.append(
            f"n={sample_size:,}: {np.median([_oracle_focal(record, 'cvae')['mae'] for record in group]):.4f}"
        )
        baseline_parts.append(
            f"{np.median([_oracle_focal(record, 'independent_softmax')['mae'] for record in group]):.4f}"
        )
    lines.extend(
        [
            "Pooled descriptively across prevalence and K, median CVAE focal MAE was "
            + "; ".join(n_parts)
            + ". The correctly specified softmax baseline improved monotonically over "
            + "the same sample sizes ("
            + " -> ".join(baseline_parts)
            + "), whereas the frozen CVAE worsened again at n=8,000. More data did not "
            "reliably cure this workflow.",
            "",
            "### Outcome cardinality",
            "",
            (
                "Focal MAE alone is misleading at K=20. The DGP holds the anchor focal "
                "probability function fixed across K and makes its nonfocal levels "
                "exchangeable; the context outcomes carry distinct class surfaces. "
                "The all-outcome probability vector, measured by equal-outcome total "
                "variation and KL regret, generally deteriorated as K increased."
            ),
            "",
            *_core_cardinality_table(core),
            "",
            "### Mixed versus uniform cardinality",
            "",
        ]
    )
    heterogeneous = [
        record
        for record in records
        if record["factors"]["design"] == "heterogeneity"
    ]
    if heterogeneous:
        comparisons = []
        for sample_size in _unique_sorted(
            [int(record["factors"]["n_train"]) for record in heterogeneous]
        ):
            group = [
                record
                for record in heterogeneous
                if int(record["factors"]["n_train"]) == sample_size
            ]
            by_variant = {
                variant: [
                    record
                    for record in group
                    if str(record["factors"]["variant"]) == variant
                ]
                for variant in ("homogeneous", "heterogeneous")
            }
            comparisons.append(
                f"n={sample_size:,}: focal MAE "
                f"{np.mean([_oracle_focal(record, 'cvae')['mae'] for record in by_variant['homogeneous']]):.4f}/"
                f"{np.mean([_oracle_focal(record, 'cvae')['mae'] for record in by_variant['heterogeneous']]):.4f} "
                "and TV "
                f"{np.mean([_oracle_summary(record, 'cvae')['mean_outcome_total_variation'] for record in by_variant['homogeneous']]):.4f}/"
                f"{np.mean([_oracle_summary(record, 'cvae')['mean_outcome_total_variation'] for record in by_variant['heterogeneous']]):.4f}"
            )
        lines.extend(
            [
                "For homogeneous `[5,5,5,5]` versus heterogeneous `[2,3,5,10]` "
                "(reported homogeneous/heterogeneous), "
                + "; ".join(comparisons)
                + ". There was no stable additional heterogeneity penalty, but all six "
                "cells failed and the designs were not entropy matched. That result is "
                "inconclusive, not reassuring.",
                "",
            ]
        )
    return lines


def _engineering_gate_summary(
    records: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
) -> List[str]:
    predictive_names = [
        name
        for name, check in records[0]["engineering_checks"]["checks"].items()
        if check["kind"] == "predictive"
    ]
    mean_predictive_passes = sum(
        all(bool(status["checks"][name]) for name in predictive_names)
        for status in statuses
    )
    predictive_fit_passes = sum(
        bool(record["engineering_checks"]["predictive_passed"]) for record in records
    )
    prerequisite_fit_passes = sum(
        bool(record["engineering_checks"]["prerequisites_passed"]) for record in records
    )
    complete_fit_passes = sum(
        bool(record["engineering_checks"]["passed"]) for record in records
    )
    labels = {
        "focal_mae": "focal MAE",
        "absolute_focal_bias": "absolute focal bias",
        "focal_p95_absolute_error": "focal p95 absolute error",
        "mean_outcome_total_variation": "mean-outcome TV",
        "mean_outcome_kl_regret": "mean-outcome KL regret",
        "quadrature_mean_absolute_change": "GH mean absolute change",
        "quadrature_p99_absolute_change": "GH p99 absolute change",
        "no_invalid_or_floor_hit": "valid probabilities / no floor hit",
    }
    lines = [
        (
            f"Only {predictive_fit_passes}/{len(records)} base fits met all five "
            f"predictive tolerances; {prerequisite_fit_passes}/{len(records)} met all "
            f"numerical/validity prerequisites; and {complete_fit_passes}/{len(records)} "
            "met both. At the cell-mean level, "
            f"{mean_predictive_passes}/{len(statuses)} cells met all five predictive "
            "checks even before requiring every seed replicate to pass."
        ),
        "",
        "| check | base fits passing | cell aggregates passing |",
        "|---|---:|---:|",
    ]
    for name in records[0]["engineering_checks"]["checks"]:
        fit_passes = sum(
            bool(record["engineering_checks"]["checks"][name]["passed"])
            for record in records
        )
        cell_passes = sum(bool(status["checks"][name]) for status in statuses)
        lines.append(
            f"| {labels.get(name, name)} | {fit_passes}/{len(records)} | "
            f"{cell_passes}/{len(statuses)} |"
        )
    lines.extend(
        [
            "",
            (
                "The cell bias gate averages the absolute seed-level biases. The "
                "low-prevalence table below instead shows mean signed bias, so opposite "
                "seed biases can cancel there and must not be used to reconstruct the gate."
            ),
        ]
    )
    return lines


def _cell_status_table(statuses: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| scenario | n | target prevalence | cardinalities / K | replicates passing | cell envelope | mean focal MAE | worst focal MAE | mean TV | mean KL regret | worst GH p99 change |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for status in sorted(
        statuses,
        key=lambda item: (
            str(item["factors"]["design"]),
            int(item["factors"]["n_train"]),
            float(item["factors"]["focal_prevalence"]),
            int(item["factors"]["maximum_cardinality"]),
            str(item["scenario"]),
        ),
    ):
        factors = status["factors"]
        cardinality = (
            str(factors["homogeneous_cardinality"])
            if factors.get("homogeneous_cardinality") is not None
            else (
                f"heterogeneous; sd(K)={float(factors['cardinality_heterogeneity']):.3f}, "
                f"sum(K)={factors['summed_cardinality']}"
            )
        )
        replicate_passes = int(status.get("replicate_pass_count", 0))
        n_replicates = int(status["n_replicates"])
        worst = status.get("worst_replicate_values", {})
        lines.append(
            f"| `{status['scenario']}` | {factors['n_train']} | "
            f"{float(factors['focal_prevalence']):g} | {cardinality} | "
            f"{replicate_passes}/{n_replicates} | "
            f"{'met' if status['passed'] else 'not met'} | "
            f"{_format(status['values']['focal_mae'])} | "
            f"{_format(worst.get('focal_mae'))} | "
            f"{_format(status['values']['mean_outcome_total_variation'])} | "
            f"{_format(status['values']['mean_outcome_kl_regret'])} | "
            f"{_format(status['values']['quadrature_p99_absolute_change'])} |"
        )
    return lines


def _mean_interval_summary(
    values: Sequence[float], digits: int = 4, *, lower_bound: Optional[float] = None
) -> str:
    mean, half_width = _mean_95_half_width(values)
    lower = mean - half_width
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    return f"{mean:.{digits}f} [{lower:.{digits}f}, {mean + half_width:.{digits}f}]"


def _optional_interval_summary(values: Sequence[Optional[float]]) -> str:
    numeric = [float(value) for value in values if value is not None]
    return _mean_interval_summary(numeric) if numeric else "NA"


def _focal_discrimination_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    core = [record for record in records if record["factors"]["design"] == "core"]
    if not core:
        return ["No core cells were included."]
    groups: Dict[Tuple[float, int, int], List[Mapping[str, Any]]] = {}
    for record in core:
        key = (
            float(record["factors"]["focal_prevalence"]),
            int(record["factors"]["n_train"]),
            int(record["factors"]["maximum_cardinality"]),
        )
        groups.setdefault(key, []).append(record)
    lines = [
        "| target prevalence | n | K | observed focal prevalence (AP null) | CVAE focal ROC-AUC | CVAE focal AP | softmax focal ROC-AUC | softmax focal AP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (prevalence, sample_size, cardinality), group in sorted(groups.items()):
        observed = [
            int(record["focal_test_count"]) / int(record["factors"]["n_test"])
            for record in group
        ]
        values: Dict[Tuple[str, str], List[Optional[float]]] = {}
        for model in CANDIDATE_MODELS:
            for key in ("auc", "pr_auc"):
                values[(model, key)] = [
                    (
                        None
                        if _focal_diagnostic(record, model).get(key) is None
                        else float(_focal_diagnostic(record, model)[key])
                    )
                    for record in group
                ]
        lines.append(
            f"| {prevalence:g} | {sample_size} | {cardinality} | "
            f"{_mean_interval_summary(observed)} | "
            f"{_optional_interval_summary(values[('cvae', 'auc')])} | "
            f"{_optional_interval_summary(values[('cvae', 'pr_auc')])} | "
            f"{_optional_interval_summary(values[('independent_softmax', 'auc')])} | "
            f"{_optional_interval_summary(values[('independent_softmax', 'pr_auc')])} |"
        )
    return lines


def _low_prevalence_count_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    selected = [
        record
        for record in records
        if record["factors"]["design"] == "core"
        and float(record["factors"]["focal_prevalence"]) <= 0.05
    ]
    if not selected:
        return ["No low-prevalence core cells were included."]
    groups: Dict[Tuple[float, int, int], List[Mapping[str, Any]]] = {}
    for record in selected:
        key = (
            float(record["factors"]["focal_prevalence"]),
            int(record["factors"]["n_train"]),
            int(record["factors"]["maximum_cardinality"]),
        )
        groups.setdefault(key, []).append(record)
    lines = [
        "| target prevalence | n | K | focal training events, median [range] | CVAE focal MAE, mean [95% t interval] | CVAE focal bias, mean [95% t interval] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (prevalence, sample_size, cardinality), group in sorted(groups.items()):
        counts = np.asarray([int(record["focal_training_count"]) for record in group])
        mae = [float(_oracle_focal(record, "cvae")["mae"]) for record in group]
        bias = [float(_oracle_focal(record, "cvae")["bias"]) for record in group]
        lines.append(
            f"| {prevalence:g} | {sample_size} | {cardinality} | "
            f"{np.median(counts):.0f} [{np.min(counts)}, {np.max(counts)}] | "
            f"{_mean_interval_summary(mae, lower_bound=0.0)} | "
            f"{_mean_interval_summary(bias)} |"
        )
    return lines


def _heterogeneity_paired_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    selected = [
        record for record in records if record["factors"]["design"] == "heterogeneity"
    ]
    if not selected:
        return ["No matched cardinality-heterogeneity cells were included."]
    specifications: Tuple[Tuple[str, MetricGetter], ...] = (
        ("focal RMSE", _oracle_focal_metric("rmse")),
        ("mean TV", _oracle_summary_metric("mean_outcome_total_variation")),
        ("oracle KL regret", _oracle_summary_metric("mean_outcome_kl_regret")),
        (
            "expected Brier regret",
            _oracle_summary_metric("mean_outcome_expected_brier_regret"),
        ),
    )
    lines = [
        "| n | model | metric | heterogeneous - homogeneous, mean [95% paired t interval] | paired seeds |",
        "|---:|---|---|---:|---:|",
    ]
    for sample_size in _unique_sorted(
        [int(record["factors"]["n_train"]) for record in selected]
    ):
        homogeneous = {
            int(record["data_seed"]): record
            for record in selected
            if int(record["factors"]["n_train"]) == sample_size
            and str(record["factors"]["variant"]) == "homogeneous"
        }
        heterogeneous = {
            int(record["data_seed"]): record
            for record in selected
            if int(record["factors"]["n_train"]) == sample_size
            and str(record["factors"]["variant"]) == "heterogeneous"
        }
        common = sorted(set(homogeneous).intersection(heterogeneous))
        if not common:
            continue
        for model in CANDIDATE_MODELS:
            for metric_label, getter in specifications:
                differences = [
                    float(getter(heterogeneous[seed], model))
                    - float(getter(homogeneous[seed], model))
                    for seed in common
                ]
                lines.append(
                    f"| {sample_size} | {MODEL_LABELS[model]} | {metric_label} | "
                    f"{_mean_interval_summary(differences)} | {len(common)} |"
                )
    return lines


def _initialization_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    sensitivity_scenarios = sorted(
        {
            str(record["scenario"])
            for record in records
            if int(record["initialization_replicate"]) > 0
        }
    )
    if not sensitivity_scenarios:
        return ["No initialization-sensitivity replicates were included in this run."]
    lines = [
        "| scenario | data seed | initialization replicate | focal MAE | focal RMSE | focal p95 error | mean TV | mean KL regret | best epoch | active latent units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in sensitivity_scenarios:
        extras = [
            record
            for record in records
            if str(record["scenario"]) == scenario
            and int(record["initialization_replicate"]) > 0
        ]
        data_seed = int(extras[0]["data_seed"])
        selected = sorted(
            [
                record
                for record in records
                if str(record["scenario"]) == scenario
                and int(record["data_seed"]) == data_seed
            ],
            key=lambda record: int(record["initialization_replicate"]),
        )
        for record in selected:
            focal = _oracle_focal(record, "cvae")
            summary = _oracle_summary(record, "cvae")
            training = record.get("cvae_training", {})
            lines.append(
                f"| `{scenario}` | {data_seed} | {record['initialization_replicate']} | "
                f"{_format(focal['mae'])} | {_format(focal['rmse'])} | "
                f"{_format(focal['absolute_error_p95'])} | "
                f"{_format(summary['mean_outcome_total_variation'])} | "
                f"{_format(summary['mean_outcome_kl_regret'])} | "
                f"{training.get('best_epoch', 'NA')} | "
                f"{training.get('best_active_latent_units', 'NA')} |"
            )
    return lines


def _quadrature_summary_table(records: Sequence[Mapping[str, Any]]) -> List[str]:
    keys = (
        ("mean_absolute_probability_change", "mean absolute probability change"),
        ("rmse_probability_change", "RMSE probability change"),
        ("p99_absolute_probability_change", "p99 absolute probability change"),
        ("maximum_absolute_probability_change", "maximum absolute probability change"),
    )
    lines = ["| GH21-vs-GH31 diagnostic | median | minimum | maximum |", "|---|---:|---:|---:|"]
    for key, label in keys:
        values = np.asarray(
            [float(record["quadrature_convergence"][key]) for record in records]
        )
        lines.append(
            f"| {label} | {np.median(values):.6f} | {np.min(values):.6f} | "
            f"{np.max(values):.6f} |"
        )
    return lines


def render_categorical_probability_report(
    run: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]],
    *,
    report_parent: Path,
) -> str:
    """Render the static Markdown body after figures have been generated."""

    validate_probability_run(run)
    base = _base_records(run)
    statuses = run["cell_engineering_status"]
    met = sum(bool(status["passed"]) for status in statuses)
    complete = bool(run["is_complete_canonical_grid"])
    sections: Dict[str, List[Mapping[str, Any]]] = {}
    for figure in figures:
        sections.setdefault(str(figure["section"]), []).append(figure)

    lines: List[str] = [
        "# Categorical CVAE conditional-probability validation",
        "",
        f"Report format: `{REPORT_VERSION}`<br>",
        f"Experiment protocol: `{run['protocol']}`<br>",
        f"Protocol SHA-256: `{run['protocol_manifest_sha256']}`",
        "",
        (
            "**Run status: complete canonical grid.**"
            if complete
            else "**Run status: partial/noncanonical subset; do not interpret it as the complete experiment.**"
        ),
        "",
        "## The question in plain English",
        "",
        (
            "For a person with covariates X, how close is the model's predicted "
            "probability for each category to the probability that actually generated "
            "the simulation? The known DGP probabilities let us answer this directly, "
            "rather than relying only on whether a single realized outcome was right."
        ),
        "",
        _plain_english_summary(base),
        "",
        (
            f"The CVAE met its regime-specific engineering envelope in {met}/{len(statuses)} "
            "evaluated cells. Within this declared experiment, that is a failed validation, "
            "not merely an inconclusive result. It does not prove that every possible CVAE "
            "workflow will fail, but the present frozen workflow should not be treated as "
            "an accurate conditional-probability engine."
        ),
        "",
        (
            "**Pipeline decision:** the categorical input/output and serialization contracts "
            "can support reversible integration and shadow runs. Do not make the current "
            "fitted CVAE a required scientific or production dependency; keep the probability "
            "model swappable while the fitting objective and marginal integration are hardened."
        ),
        "",
        (
            "Focused remediation is tracked in [GitHub Issue #3]"
            "(https://github.com/jarrod-dalton/multioutcome-cvae/issues/3); "
            "the broader predictive-validity Issue #2 remains open."
        ),
        "",
        *_factor_finding_lines(base),
        "## How to read the evidence",
        "",
        "- `p-true` is the exact simulated conditional probability; `p-hat` is a fitted model probability.",
        "- MAE, RMSE, total variation, KL regret, and expected Brier regret directly compare p-hat with p-true; lower is better.",
        "- Held-out NLL and multiclass Brier use realized outcomes and are proper predictive scores; lower is better.",
        "- ROC-AUC and average precision (PR-AUC/AP) measure ranking, not calibration. AP is especially useful for rare classes, but its baseline changes with prevalence.",
        "- AUC/AP are undefined when a test class has no positive or no negative observation. Coverage is shown explicitly; partial defined-class macros are not complete evidence.",
        "- Trend-plot bars are two-sided 95% Student-t intervals across the fixed data seeds within each exact core cell unless a caption says they are full ranges. Displayed lower whiskers for nonnegative metrics are truncated at zero.",
        "",
        "## Descriptive comparison over included base fits",
        "",
        "Entries are median [minimum, maximum] across fit records; regimes are intentionally not pooled into one inferential estimate. Raw held-out multiclass Brier scores are comparable only within the same schema/cell because their scale changes with cardinality and outcome entropy; direct regret is the safer cross-regime quantity.",
        "",
        *_model_descriptive_table(base),
        "",
    ]

    if "Core factorial results" in sections:
        lines.extend(
            [
                "## Core factorial results",
                "",
                "### Low-prevalence cells in terms of events actually observed during training",
                "",
                (
                    "The MAE interval is truncated at its natural lower bound of zero. "
                    "Signed bias can cancel across seeds; the engineering gate instead "
                    "uses the mean absolute seed-level bias."
                ),
                "",
                *_low_prevalence_count_table(base),
                "",
                "### Focal-class ROC-AUC and average precision",
                "",
                (
                    "Average precision must be interpreted against the observed focal "
                    "test prevalence shown in the same row, not against a universal 0.5 baseline."
                ),
                "",
                *_focal_discrimination_table(base),
                "",
            ]
        )
        for figure in sections["Core factorial results"]:
            lines.extend(
                [
                    f"### {figure['caption']}",
                    "",
                    _markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])),
                    "",
                ]
            )

    lines.extend(
        [
            "## Regime-specific engineering envelopes",
            "",
            (
                "Each row is one exact DGP/training regime. `met` requires both the "
                "cell-mean checks and every fixed-seed replicate to meet the declared "
                "fit-level checks; it is not permission to extrapolate to another regime."
            ),
            "",
            *_engineering_gate_summary(base, statuses),
            "",
            *_cell_status_table(statuses),
            "",
        ]
    )

    if "Cardinality heterogeneity" in sections:
        lines.extend(
            [
                "## Cardinality heterogeneity",
                "",
                (
                    "The homogeneous `[5,5,5,5]` and heterogeneous `[2,3,5,10]` "
                    "designs have the same number of semantic outcomes and the same "
                    "summed decoder width. They are **not entropy matched**, so any "
                    "difference cannot be attributed to cardinality heterogeneity alone. "
                    "Within n and seed, both variants share X and the anchor's p-true/Y, "
                    "so the table and plot use paired heterogeneous-minus-homogeneous differences."
                ),
                "",
                *_heterogeneity_paired_table(base),
                "",
            ]
        )
        for figure in sections["Cardinality heterogeneity"]:
            lines.extend(
                [
                    _markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])),
                    "",
                ]
            )

    if "Dependence control" in sections:
        lines.extend(
            [
                "## Residual-dependence control",
                "",
                "This comparison holds the marginal conditional probabilities fixed while changing the Gaussian-copula residual dependence.",
                "",
            ]
        )
        for figure in sections["Dependence control"]:
            lines.extend(
                [_markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])), ""]
            )

    quadrature_passes = sum(
        bool(record["engineering_checks"]["prerequisites_passed"])
        for record in base
    )
    lines.extend(
        [
            "## Numerical integration diagnostics",
            "",
            (
                f"Only {quadrature_passes}/{len(base)} base fits met all numerical/validity "
                "prerequisites. All probability matrices were syntactically valid, so the "
                "failures came from GH21-versus-GH31 disagreement. Some large-n error "
                "magnitudes are therefore numerically uncertain; predictive checks also "
                "failed in stable-integration cells, so this does not explain away the "
                "probability-recovery failure."
            ),
            "",
            *_quadrature_summary_table(base),
            "",
            (
                "A post-run audit found that the original undamped Newton solver for the "
                "experimental `truth_calibration_intercept`/`slope` fields could diverge "
                "for rare classes. The committed artifact is a disclosed correction-only "
                "rerun using centered/scaled damped Newton updates. Every nonexcluded result "
                "field matched the first complete run exactly. These coefficients remain "
                "excluded from the gates and headline evidence."
            ),
            "",
        ]
    )
    for figure in sections.get("Numerical diagnostics", []):
        lines.extend(
            [_markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])), ""]
        )

    lines.extend(["## Initialization sensitivity", "", *_initialization_table(run["records"]), ""])
    for figure in sections.get("Initialization sensitivity", []):
        lines.extend(
            [_markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])), ""]
        )

    if "Sentinel probability and calibration plots" in sections:
        lines.extend(
            [
                "## Individual probability and calibration plots",
                "",
                (
                    "Sentinels were fixed before fitting. Agreement panels plot individual "
                    "p-hat values directly against p-true using the capped plotting rows. "
                    "Reliability panels use the full held-out test set, bin on each "
                    "candidate model's p-hat and show both the observed event fraction "
                    "(95% Wilson interval) and mean p-true in that same bin. Bin counts are "
                    "summarized in each panel subtitle. Low-probability panels restrict the "
                    "predicted-probability x axis but retain the full 0-1 y axis, so severe "
                    "underprediction remains visible rather than being clipped."
                ),
                "",
            ]
        )
        for figure in sections["Sentinel probability and calibration plots"]:
            lines.extend(
                [
                    f"### {figure['caption']}",
                    "",
                    _markdown_image(Path(figure["path"]), report_parent, str(figure["caption"])),
                    "",
                ]
            )

    environment = run.get("environment", {})
    lines.extend(
        [
            "## Reproducibility and provenance",
            "",
            "| item | value |",
            "|---|---|",
            f"| selected scenarios | {len(run['selected_scenarios'])} |",
            f"| selected data seeds | `{run['selected_data_seeds']}` |",
            f"| base fits | {len(base)} |",
            f"| total fits including initialization sensitivity | {len(run['records'])} |",
            f"| runtime seconds | {_format(run['runtime_seconds'], 1)} |",
            f"| device | `{run.get('device', 'unknown')}` |",
        ]
    )
    for key, value in environment.items():
        lines.append(f"| {key.replace('_', ' ')} | `{value}` |")

    development = run.get("development_history", {})
    lines.extend(
        [
            "",
            "### Development-history disclosure",
            "",
            f"- Retired proposed seed set: `{development.get('retired_proposed_seed_set', 'not recorded')}`.",
        ]
    )
    seen_checks = development.get("seen_development_checks", [])
    if seen_checks:
        lines.append("- Checks whose results were seen during development:")
        for check in seen_checks:
            lines.append(f"  - {check}")
    else:
        lines.append("- Checks whose results were seen during development: not recorded.")
    lines.extend(
        [
            f"- Disposition: {development.get('disposition', 'not recorded')}",
            f"- Canonical seed set: `{development.get('canonical_seed_set', run['selected_data_seeds'])}`.",
            "",
            "### Protocol rerun commands",
            "",
            "The machine-readable result is committed losslessly as `docs/categorical_probability_validation_results.json.gz` to avoid adding a 62 MB pretty-printed JSON file to Git history.",
            (
                "The committed correction-only canonical artifact was produced from Git commit "
                f"`{environment.get('git_head', 'not recorded')}` with runner SHA-256 "
                f"`{environment.get('runner_source_sha256', 'not recorded')}`. The first "
                "complete run came from `113c5a6080980076ce753187c3f93ed72ab86e42`; "
                "all fields other than runtime/provenance and the corrected, excluded "
                "truth-calibration coefficients matched exactly. Checkout the recorded "
                "artifact commit to reproduce the exact canonical runner."
            ),
            "",
            "```bash",
            "PYTHONPATH=python .venv/bin/python -m validation.categorical_probability_validation --output /tmp/categorical_probability_validation_results.json --verbose",
            "gzip -n -9 /tmp/categorical_probability_validation_results.json",
            "MPLCONFIGDIR=/tmp/multioutcome-cvae-matplotlib PYTHONPATH=python .venv/bin/python -m validation.categorical_probability_report /tmp/categorical_probability_validation_results.json.gz --output docs/categorical_probability_validation.md --figure-dir docs/categorical_probability_validation_figures",
            "```",
            "",
            "### Exact protocol manifest",
            "",
            "```json",
            json.dumps(run["protocol_manifest"], indent=2, sort_keys=True),
            "```",
            "",
            "## Limits of this experiment",
            "",
            "- The DGP uses bounded covariates and model-aligned linear-softmax marginals. It tests interpolation and controlled recovery, not arbitrary real-world misspecification or extrapolation.",
            "- The sample-size, prevalence, cardinality, architecture, optimization, and dependence settings are the declared grid; results do not automatically transfer beyond it.",
            "- The decoder-width-matched heterogeneity comparison has equal J and sum(K), but it is not entropy matched.",
            "- The anchor focal probability function is held fixed across K and its nonfocal levels are exchangeable; context outcomes have distinct class surfaces. This isolates focal recovery but makes focal MAE alone an incomplete high-K diagnostic.",
            "- Reliability from realized outcomes is noisy for rare classes even with a large test set; direct oracle errors are the primary simulation evidence.",
            "- Initialization replicates reuse a fixed data split and therefore measure optimization sensitivity, not new-sample uncertainty.",
            "- Quadrature comparisons assess GH21 versus GH31 agreement, not mathematical proof that either order is exact.",
            "- Experimental truth-calibration intercept/slope fields were corrected after a disclosed rare-class solver defect and remain excluded from the gates and headline evidence.",
            "- Engineering envelopes are regime-specific safeguards rather than a global certification of predictive validity.",
            "",
        ]
    )
    return "\n".join(lines)


def write_categorical_probability_report(
    run: Mapping[str, Any],
    output_path: Path = Path("docs/categorical_probability_validation.md"),
    *,
    figure_dir: Optional[Path] = None,
    dpi: int = 150,
) -> Path:
    """Generate PNGs and write the compiled Markdown report."""

    validate_probability_run(run)
    output_path = Path(output_path)
    if figure_dir is None:
        figure_dir = output_path.parent / f"{output_path.stem}_figures"
    figures = generate_probability_report_figures(run, Path(figure_dir), dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_categorical_probability_report(
            run, figures, report_parent=output_path.parent
        ),
        encoding="utf-8",
    )
    return output_path.resolve()


def load_probability_run(path: Path) -> Dict[str, Any]:
    path = Path(path)
    handle_context = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with handle_context as handle:
        run = json.load(handle)
    validate_probability_run(run)
    return run


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="JSON written by the validation runner, optionally gzip-compressed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/categorical_probability_validation.md"),
        help="Compiled Markdown output path.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="Optional PNG directory (defaults beside the Markdown file).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    arguments = parser.parse_args(argv)
    run = load_probability_run(arguments.results)
    written = write_categorical_probability_report(
        run,
        arguments.output,
        figure_dir=arguments.figure_dir,
        dpi=arguments.dpi,
    )
    print(f"Wrote {written}")
    return 0


__all__ = [
    "REPORT_VERSION",
    "generate_probability_report_figures",
    "load_probability_run",
    "render_categorical_probability_report",
    "validate_probability_run",
    "write_categorical_probability_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
