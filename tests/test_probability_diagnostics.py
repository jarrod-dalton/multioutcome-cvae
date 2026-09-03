import importlib
from pathlib import Path

import numpy as np
import pytest

from validation.probability_diagnostics import (
    binary_auc,
    binary_pr_auc,
    calibration_curve,
    compare_probability_models,
    experiment_metric_record,
    plot_calibration_pages,
    plot_metric_trends,
    plot_oracle_agreement_pages,
    probability_diagnostics,
    render_probability_diagnostics_markdown,
    render_probability_comparison_markdown,
    write_probability_comparison_bundle,
)


probability_module = importlib.import_module("validation.probability_diagnostics")


SCHEMA = (
    {"name": "binary", "levels": ("no", "yes")},
    {"name": "three", "levels": ("a", "b", "c")},
)


def _example_inputs():
    y = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 2],
            [1, 0],
            [0, 1],
            [1, 2],
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )
    oracle = {
        "binary": np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.30, 0.70],
                [0.20, 0.80],
                [0.70, 0.30],
                [0.10, 0.90],
                [0.60, 0.40],
                [0.40, 0.60],
            ]
        ),
        "three": np.array(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.70, 0.10],
                [0.10, 0.20, 0.70],
                [0.60, 0.30, 0.10],
                [0.20, 0.60, 0.20],
                [0.10, 0.30, 0.60],
                [0.65, 0.25, 0.10],
                [0.15, 0.65, 0.20],
            ]
        ),
    }
    fitted = {
        name: 0.85 * matrix + 0.15 / matrix.shape[1] for name, matrix in oracle.items()
    }
    intercept = {
        "binary": np.tile(np.array([[0.5, 0.5]]), (y.shape[0], 1)),
        "three": np.tile(np.array([[0.375, 0.375, 0.25]]), (y.shape[0], 1)),
    }
    return y, oracle, fitted, intercept


def test_auc_and_pr_auc_have_explicit_tie_behavior():
    observed = np.array([0, 0, 1, 1])
    probability = np.array([0.10, 0.40, 0.35, 0.80])
    assert binary_auc(observed, probability) == pytest.approx(0.75)
    assert binary_pr_auc(observed, probability) == pytest.approx(5.0 / 6.0)

    tied_y = np.array([1, 0])
    tied_probability = np.array([0.5, 0.5])
    assert binary_auc(tied_y, tied_probability) == pytest.approx(0.5)
    assert binary_pr_auc(tied_y, tied_probability) == pytest.approx(0.5)
    assert binary_auc(np.ones(3), np.arange(3)) is None
    assert binary_pr_auc(np.zeros(3), np.arange(3)) is None


def test_diagnostics_share_one_vectorized_ranking_sort_per_class(monkeypatch):
    y, oracle, fitted, _ = _example_inputs()
    original_argsort = probability_module.np.argsort
    calls = []

    def counted_argsort(*args, **kwargs):
        calls.append(1)
        return original_argsort(*args, **kwargs)

    monkeypatch.setattr(probability_module.np, "argsort", counted_argsort)
    probability_diagnostics(
        fitted,
        y,
        SCHEMA,
        oracle_probabilities=oracle,
        n_bins=4,
    )
    assert len(calls) == sum(len(entry["levels"]) for entry in SCHEMA)


def test_quantile_calibration_collapses_constant_predictions_safely():
    curve = calibration_curve(np.array([0, 1, 0, 1]), np.repeat(0.25, 4), n_bins=10)
    assert len(curve) == 1
    assert curve[0]["count"] == 4
    assert curve[0]["mean_predicted"] == pytest.approx(0.25)
    assert curve[0]["observed_fraction"] == pytest.approx(0.5)
    assert curve[0]["observed_wilson_lower_95"] < 0.5
    assert curve[0]["observed_wilson_upper_95"] > 0.5


def test_probability_diagnostics_cover_observed_and_oracle_targets():
    y, oracle, fitted, _ = _example_inputs()
    diagnostics = probability_diagnostics(
        fitted,
        y,
        SCHEMA,
        oracle_probabilities=oracle,
        n_bins=4,
    )
    assert diagnostics["summary"]["n"] == 8
    assert diagnostics["summary"]["n_outcomes"] == 2
    assert diagnostics["summary"]["n_classes"] == 5
    assert diagnostics["summary"]["macro_one_vs_rest_auc"] > 0.5
    assert diagnostics["summary"]["macro_one_vs_rest_pr_auc"] > 0.5
    assert diagnostics["summary"]["oracle_probability_rmse"] > 0.0
    assert diagnostics["summary"]["oracle_probability_rmse"] < 0.1
    assert {record["outcome"] for record in diagnostics["outcomes"]} == {
        "binary",
        "three",
    }
    assert diagnostics["true_probability_bands"]
    assert any(
        record["lower"] == 0.10 and record["upper"] == 0.20
        for record in diagnostics["true_probability_bands"]
    )
    expected_equal_outcome_mae = np.mean(
        [record["oracle_probability_mae"] for record in diagnostics["outcomes"]]
    )
    expected_equal_outcome_rmse = np.sqrt(
        np.mean(
            [
                record["oracle_probability_rmse"] ** 2
                for record in diagnostics["outcomes"]
            ]
        )
    )
    assert diagnostics["summary"]["oracle_probability_mae"] == pytest.approx(
        expected_equal_outcome_mae
    )
    assert diagnostics["summary"]["oracle_probability_rmse"] == pytest.approx(
        expected_equal_outcome_rmse
    )
    assert "pooled_class_cell_oracle_probability_rmse" in diagnostics["summary"]
    assert diagnostics["summary"]["mean_outcome_total_variation"] > 0.0
    assert diagnostics["summary"]["mean_outcome_oracle_kl"] > 0.0
    assert diagnostics["summary"]["mean_outcome_brier_regret"] > 0.0
    for record in diagnostics["classes"]:
        assert "auc" in record
        assert "pr_auc" in record
        assert "binary_brier" in record
        assert "ece" in record
        assert "oracle_probability_mae" in record
        assert all(
            "mean_oracle_probability" in bin_record
            for bin_record in record["reliability_bins"]
        )


def test_oracle_kl_and_brier_regret_are_exact_proper_score_regrets():
    schema = ({"name": "outcome", "levels": ("a", "b", "c")},)
    y = np.array([[0], [2]], dtype=np.int32)
    oracle = {
        "outcome": np.array([[0.20, 0.30, 0.50], [0.00, 0.25, 0.75]], dtype=np.float64)
    }
    fitted = {
        "outcome": np.array([[0.10, 0.40, 0.50], [0.20, 0.30, 0.50]], dtype=np.float64)
    }
    diagnostics = probability_diagnostics(
        fitted,
        y,
        schema,
        oracle_probabilities=oracle,
        n_bins=2,
        epsilon=0.10,
    )
    q = oracle["outcome"]
    p = fitted["outcome"]
    positive = q > 0.0
    expected_kl = np.sum(q[positive] * np.log(q[positive] / p[positive])) / len(y)
    expected_brier_regret = np.mean(np.sum(np.square(p - q), axis=1))
    outcome = diagnostics["outcomes"][0]
    assert outcome["oracle_kl"] == pytest.approx(expected_kl)
    assert outcome["oracle_kl"] >= 0.0
    assert outcome["brier_regret"] == pytest.approx(expected_brier_regret)

    identity = probability_diagnostics(
        oracle,
        y,
        schema,
        oracle_probabilities=oracle,
        n_bins=2,
        epsilon=0.10,
    )
    assert identity["outcomes"][0]["oracle_kl"] == pytest.approx(0.0)
    assert identity["outcomes"][0]["brier_regret"] == pytest.approx(0.0)


def test_oracle_kl_retains_infinite_regret_and_discloses_probability_floors():
    schema = ({"name": "binary", "levels": ("no", "yes")},)
    y = np.array([[0], [1]], dtype=np.int32)
    oracle = {"binary": np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)}
    fitted = {"binary": np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float64)}
    diagnostics = probability_diagnostics(
        fitted,
        y,
        schema,
        oracle_probabilities=oracle,
        n_bins=2,
    )
    outcome = diagnostics["outcomes"][0]
    summary = diagnostics["summary"]
    assert np.isinf(outcome["oracle_kl"])
    assert np.isinf(summary["mean_outcome_oracle_kl"])
    assert summary["positive_oracle_at_fitted_zero"] == 1
    assert summary["fitted_probability_floor_hits"] == 1
    assert summary["selected_probability_floor_hits"] == 1
    assert summary["oracle_probability_floor_hits"] == 1


def test_incomplete_ranking_metrics_are_not_silently_reported_as_complete():
    schema = ({"name": "three", "levels": ("a", "b", "c")},)
    y = np.zeros((4, 1), dtype=np.int32)
    fitted = {
        "three": np.array(
            [
                [0.80, 0.10, 0.10],
                [0.70, 0.20, 0.10],
                [0.60, 0.20, 0.20],
                [0.50, 0.25, 0.25],
            ]
        )
    }
    summary = probability_diagnostics(fitted, y, schema, n_bins=2)["summary"]
    assert summary["macro_one_vs_rest_auc"] is None
    assert summary["partial_macro_one_vs_rest_auc"] is None
    assert summary["one_vs_rest_auc_defined_classes"] == 0
    assert summary["one_vs_rest_auc_required_classes"] == 3
    assert summary["one_vs_rest_auc_coverage"] == pytest.approx(0.0)
    assert summary["macro_one_vs_rest_pr_auc"] is None
    assert summary["partial_macro_one_vs_rest_pr_auc"] == pytest.approx(1.0)
    assert summary["one_vs_rest_pr_auc_defined_classes"] == 1
    assert summary["one_vs_rest_pr_auc_required_classes"] == 3
    assert summary["one_vs_rest_pr_auc_coverage"] == pytest.approx(1.0 / 3.0)


def test_oracle_summaries_weight_semantic_outcomes_not_class_cells():
    schema = (
        {"name": "two", "levels": ("a", "b")},
        {"name": "four", "levels": ("a", "b", "c", "d")},
    )
    y = np.array([[0, 0], [1, 1], [0, 2], [1, 3]], dtype=np.int32)
    oracle = {
        "two": np.tile(np.array([[0.50, 0.50]]), (4, 1)),
        "four": np.tile(np.array([[0.25, 0.25, 0.25, 0.25]]), (4, 1)),
    }
    fitted = {
        "two": np.tile(np.array([[0.60, 0.40]]), (4, 1)),
        "four": np.tile(np.array([[0.45, 0.25, 0.15, 0.15]]), (4, 1)),
    }
    diagnostics = probability_diagnostics(
        fitted,
        y,
        schema,
        oracle_probabilities=oracle,
        n_bins=2,
    )
    outcomes = diagnostics["outcomes"]
    summary = diagnostics["summary"]
    expected_kl = np.mean([record["oracle_kl"] for record in outcomes])
    expected_regret = np.mean([0.02, 0.06])
    pooled_cell_mse = (0.02 + 0.06) / 6.0
    assert summary["mean_outcome_oracle_kl"] == pytest.approx(expected_kl)
    assert summary["mean_outcome_brier_regret"] == pytest.approx(expected_regret)
    assert summary["mean_outcome_brier_regret"] != pytest.approx(pooled_cell_mse)
    assert summary["oracle_probability_mean_absolute_class_bias"] == pytest.approx(0.10)
    assert summary["oracle_probability_rms_class_bias"] == pytest.approx(
        np.sqrt((0.01 + 0.015) / 2.0)
    )
    assert summary["oracle_probability_maximum_absolute_class_bias"] == pytest.approx(
        0.20
    )
    assert "oracle_probability_bias" not in summary


def test_quantile_calibration_does_not_split_equal_scores_across_bins():
    observed = np.array([0, 1, 0, 1, 0, 1])
    predicted = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.9])
    curve = calibration_curve(observed, predicted, n_bins=4)
    assert sum(record["count"] for record in curve) == len(observed)
    assert all(
        left["maximum_predicted"] < right["minimum_predicted"]
        for left, right in zip(curve, curve[1:])
    )


def test_named_model_comparison_and_markdown_contract(tmp_path):
    y, oracle, fitted, intercept = _example_inputs()
    comparison = compare_probability_models(
        {"CVAE": fitted, "independent softmax": intercept},
        y,
        SCHEMA,
        oracle_probabilities=oracle,
        n_bins=4,
    )
    assert set(comparison["models"]) == {"CVAE", "independent softmax"}
    assert comparison["has_oracle"] is True
    record = experiment_metric_record(
        comparison["models"]["CVAE"],
        "CVAE",
        {
            "n_train": 500,
            "focal_prevalence": 0.05,
            "maximum_cardinality": 5,
            "cardinality_heterogeneity": 3,
        },
        scenario="small_rare",
        seed=101,
    )
    assert record["n_train"] == 500
    assert record["oracle_probability_rmse"] > 0.0

    markdown = render_probability_comparison_markdown(
        comparison,
        [(tmp_path / "figure.png", "Example figure")],
        markdown_parent=tmp_path,
    )
    assert "independent softmax" in markdown
    assert "PR-AUC" in markdown
    assert "![Example figure](figure.png)" in markdown


def test_markdown_escapes_dynamic_table_cells_and_image_links(tmp_path):
    schema = (
        {
            "name": "out|come\ncontinued",
            "levels": ("no|zero", "yes\nindeed"),
        },
    )
    y = np.array([[0], [1], [0], [1]], dtype=np.int32)
    fitted = {
        "out|come\ncontinued": np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8]]
        )
    }
    diagnostics = probability_diagnostics(fitted, y, schema, n_bins=2)
    markdown = render_probability_diagnostics_markdown(
        diagnostics,
        [(tmp_path / "figure with space.png", "Caption]\ncontinued")],
        markdown_parent=tmp_path,
        title="Title\ncontinued",
    )
    assert markdown.startswith("# Title continued")
    assert "out\\|come continued" in markdown
    assert "no\\|zero" in markdown
    assert "yes indeed" in markdown
    assert "![Caption\\] continued](<figure with space.png>)" in markdown

    comparison = compare_probability_models(
        {"model|name\ncontinued": fitted}, y, schema, n_bins=2
    )
    comparison_markdown = render_probability_comparison_markdown(
        comparison, [], markdown_parent=tmp_path
    )
    assert "model\\|name continued" in comparison_markdown


def test_low_probability_plots_show_overflow_and_subset_statistics(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    schema = ({"name": "binary", "levels": ("no", "yes")},)
    y = np.array([[0], [0], [1], [1]], dtype=np.int32)
    fitted = {
        "binary": np.array([[0.05, 0.95], [0.10, 0.90], [0.15, 0.85], [0.20, 0.80]])
    }
    oracle = {
        "binary": np.array([[0.10, 0.90], [0.20, 0.80], [0.30, 0.70], [0.40, 0.60]])
    }
    diagnostics = probability_diagnostics(
        fitted, y, schema, oracle_probabilities=oracle, n_bins=2
    )
    captured = []

    def capture_figure(fig, path, dpi):
        captured.append(fig)
        return Path(path)

    monkeypatch.setattr(probability_module, "_save_figure", capture_figure)
    plot_calibration_pages(
        diagnostics,
        tmp_path,
        probability_limit=0.20,
        maximum_panels_per_page=2,
    )
    calibration_figure = captured.pop()
    calibration_axis = calibration_figure.axes[0]
    assert calibration_axis.get_xlim() == pytest.approx((0.0, 0.20))
    assert calibration_axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert any(text.get_text().startswith("n=") for text in calibration_axis.texts)

    oracle_plot_fitted = {
        "binary": np.array([[0.90, 0.10], [0.80, 0.20], [0.70, 0.30], [0.60, 0.40]])
    }
    plot_oracle_agreement_pages(
        oracle_plot_fitted,
        oracle,
        schema,
        tmp_path,
        probability_limit=0.20,
        maximum_panels_per_page=2,
    )
    oracle_figure = captured.pop()
    oracle_axis = oracle_figure.axes[0]
    assert oracle_axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert "shown n=2; MAE=0.700; RMSE=0.707" in oracle_axis.get_title()

    plt = probability_module._pyplot()
    plt.close(calibration_figure)
    plt.close(oracle_figure)


def test_trend_plots_show_raw_runs_without_naive_standard_error_bars(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    from matplotlib.collections import LineCollection

    records = [
        {
            "model": model,
            "n_train": n_train,
            "oracle_probability_rmse": value,
        }
        for model, values in (("CVAE", (0.12, 0.10)), ("baseline", (0.16, 0.15)))
        for n_train, value in zip((500, 2000), values)
    ]
    captured = []

    def capture_figure(fig, path, dpi):
        captured.append(fig)
        return Path(path)

    monkeypatch.setattr(probability_module, "_save_figure", capture_figure)
    plot_metric_trends(
        records,
        tmp_path,
        factor_labels={"n_train": "training sample size"},
        metrics=(("oracle_probability_rmse", "oracle RMSE"),),
    )
    figure = captured.pop()
    assert all(
        not isinstance(collection, LineCollection)
        for axis in figure.axes
        for collection in axis.collections
    )
    assert "dots are fitted runs; lines are descriptive cell means" in (
        figure._suptitle.get_text()
    )
    probability_module._pyplot().close(figure)


def test_plot_and_bundle_smoke_when_optional_dependency_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    y, oracle, fitted, intercept = _example_inputs()
    bundle = write_probability_comparison_bundle(
        tmp_path / "bundle",
        "smoke",
        {"CVAE": fitted, "independent softmax": intercept},
        y,
        SCHEMA,
        oracle_probabilities=oracle,
        n_bins=4,
        low_probability_limit=0.2,
        maximum_panels_per_page=4,
        dpi=72,
    )
    markdown_path = Path(bundle["markdown_path"])
    assert markdown_path.is_file()
    assert "Observed calibration by model" in markdown_path.read_text()
    assert bundle["image_paths"]
    assert all(
        Path(path).is_file() and Path(path).stat().st_size > 0
        for path in bundle["image_paths"]
    )

    comparison = bundle["comparison"]
    trend_records = []
    for n_train, multiplier in ((500, 1.0), (2000, 0.8)):
        for model, diagnostics in comparison["models"].items():
            record = experiment_metric_record(
                diagnostics,
                model,
                {
                    "n_train": n_train,
                    "focal_prevalence": 0.05,
                    "maximum_cardinality": 3,
                    "cardinality_heterogeneity": 1,
                },
                seed=n_train,
            )
            # Create a visible deterministic trend without changing the metric API.
            record["oracle_probability_rmse"] *= multiplier
            trend_records.append(record)
    paths = plot_metric_trends(
        trend_records,
        tmp_path / "trends",
        factor_labels={
            "n_train": "training sample size",
            "focal_prevalence": "focal prevalence",
            "maximum_cardinality": "maximum K",
            "cardinality_heterogeneity": "max(K) - min(K)",
        },
        filename_prefix="smoke",
        dpi=72,
    )
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        {"binary": np.ones((2, 2)), "three": np.ones((2, 3)) / 3.0},
        {"binary": np.ones((2, 2)) / 2.0},
    ],
)
def test_probability_contract_rejects_malformed_mappings(bad_probabilities):
    with pytest.raises(ValueError):
        probability_diagnostics(
            bad_probabilities,
            np.zeros((2, 2), dtype=np.int32),
            SCHEMA,
        )
