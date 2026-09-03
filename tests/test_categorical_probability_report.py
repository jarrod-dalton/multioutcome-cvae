import gzip
from pathlib import Path

import numpy as np

from validation.categorical_probability_report import (
    load_probability_run,
    validate_probability_run,
    write_categorical_probability_report,
)
from validation.categorical_probability_validation import (
    aggregate_cell_engineering_status,
    compact_probability_diagnostics,
    evaluate_fit_engineering_checks,
    oracle_expected_probability_metrics,
    write_probability_results,
)
from validation.probability_diagnostics import compare_probability_models


def _tiny_report_run():
    rng = np.random.default_rng(20260902)
    rows = 160
    schema = [
        {"name": "anchor", "levels": ["common", "focal"]},
        {"name": "context", "levels": ["low", "high"]},
    ]
    anchor_true = np.linspace(0.005, 0.095, rows)
    context_true = np.linspace(0.25, 0.75, rows)
    oracle = {
        "anchor": np.column_stack((1.0 - anchor_true, anchor_true)),
        "context": np.column_stack((1.0 - context_true, context_true)),
    }

    def binary_mapping(anchor, context):
        return {
            "anchor": np.column_stack((1.0 - anchor, anchor)),
            "context": np.column_stack((1.0 - context, context)),
        }

    cvae = binary_mapping(
        np.clip(anchor_true * 1.08 + 0.004, 0.0, 1.0),
        np.clip(context_true * 0.92 + 0.04, 0.0, 1.0),
    )
    softmax = binary_mapping(
        np.clip(anchor_true * 0.98 + 0.001, 0.0, 1.0),
        np.clip(context_true * 0.97 + 0.015, 0.0, 1.0),
    )
    y = np.column_stack(
        (
            rng.binomial(1, anchor_true),
            rng.binomial(1, context_true),
        )
    ).astype(np.int32)
    # Guarantee ranking metrics are estimable in a tiny deterministic fixture.
    y[0, 0] = 1
    y[1, 0] = 0
    comparison = compare_probability_models(
        {"cvae": cvae, "independent_softmax": softmax},
        y,
        schema,
        oracle_probabilities=oracle,
        n_bins=5,
    )
    compact, focal_bands = compact_probability_diagnostics(
        comparison, anchor_outcome="anchor", focal_class_index=1
    )
    oracle_metrics = {
        model: oracle_expected_probability_metrics(
            probabilities,
            oracle,
            schema,
            anchor_outcome="anchor",
            focal_class_index=1,
            epsilon=1.0e-12,
        )
        for model, probabilities in {
            "cvae": cvae,
            "independent_softmax": softmax,
        }.items()
    }
    quadrature = {
        "mean_absolute_probability_change": 1.0e-5,
        "rmse_probability_change": 2.0e-5,
        "p99_absolute_probability_change": 5.0e-5,
        "maximum_absolute_probability_change": 8.0e-5,
    }
    validity = {
        "invalid_values": 0,
        "invalid_row_sums": 0,
        "floor_hits": 0,
    }
    engineering = evaluate_fit_engineering_checks(
        oracle_metrics["cvae"], quadrature, validity, focal_prevalence=0.05
    )
    full_test_bins = {
        model: [
            {
                "outcome": item["outcome"],
                "level": item["level"],
                "class_index": item["class_index"],
                "n": item["n"],
                "event_count": item["event_count"],
                "reliability_bins": item["reliability_bins"],
            }
            for item in comparison["models"][model]["classes"]
            if item["outcome"] == "anchor"
        ]
        for model in ("cvae", "independent_softmax")
    }
    record = {
        "scenario": "core_n500_q050_k2",
        "data_seed": 1701,
        "initialization_replicate": 0,
        "is_initialization_sensitivity": False,
        "factors": {
            "design": "core",
            "variant": "homogeneous",
            "n_train": 500,
            "n_validation": 250,
            "n_test": rows,
            "focal_prevalence": 0.05,
            "semantic_outcomes": 2,
            "minimum_cardinality": 2,
            "maximum_cardinality": 2,
            "summed_cardinality": 4,
            "cardinality_heterogeneity": 0.0,
            "homogeneous_cardinality": 2,
            "copula_rho": 0.35,
        },
        "cardinalities": [2, 2],
        "schema": schema,
        "anchor_outcome": "anchor",
        "focal_class_index": 1,
        "split_integrity": True,
        "training_counts": {"anchor": [477, 23], "context": [250, 250]},
        "focal_training_count": 23,
        "focal_test_count": int(y[:, 0].sum()),
        "cvae_training": {
            "epochs_ran": 4,
            "best_epoch": 3,
            "best_active_latent_units": 2,
        },
        "independent_softmax_training": {"converged": True},
        "model_diagnostics": compact,
        "focal_true_probability_bands": focal_bands,
        "oracle_metrics": oracle_metrics,
        "quadrature_convergence": quadrature,
        "probability_validity": validity,
        "engineering_checks": engineering,
        "runtime_seconds": 0.1,
        "plot_payload": {
            "selection": "tiny fixed test rows",
            "rows": 40,
            "Y": y[:40],
            "anchor_reliability_bins_full_test": full_test_bins,
            "oracle_probabilities": {
                name: matrix[:40] for name, matrix in oracle.items()
            },
            "model_probabilities": {
                "cvae": {name: matrix[:40] for name, matrix in cvae.items()},
                "independent_softmax": {
                    name: matrix[:40] for name, matrix in softmax.items()
                },
            },
        },
    }
    statuses = aggregate_cell_engineering_status([record])
    development_history = {
        "retired_proposed_seed_set": [4243],
        "seen_development_checks": ["A disclosed implementation-only smoke check."],
        "disposition": "Retired without changing the protocol.",
        "canonical_seed_set": [1701],
    }
    manifest = {
        "protocol": "tiny-probability-report-test",
        "development_history": development_history,
        "scenarios": [{"name": record["scenario"]}],
    }
    return {
        "protocol": "tiny-probability-report-test",
        "protocol_manifest_sha256": "0" * 64,
        "protocol_manifest": manifest,
        "development_history": development_history,
        "config": {"calibration_bins": 5, "plot_rows": rows},
        "environment": {"python": "test", "git_head": "test"},
        "device": "cpu",
        "selected_scenarios": [record["scenario"]],
        "selected_data_seeds": [1701],
        "is_complete_canonical_grid": False,
        "records": [record],
        "cell_engineering_status": statuses,
        "runtime_seconds": 0.1,
    }


def test_report_round_trip_writes_markdown_and_pngs(tmp_path):
    run = _tiny_report_run()
    validate_probability_run(run)
    json_path = write_probability_results(run, tmp_path / "results.json")
    loaded = load_probability_run(json_path)
    output = write_categorical_probability_report(
        loaded,
        tmp_path / "categorical_probability_validation.md",
        figure_dir=tmp_path / "figures",
        dpi=55,
    )
    report = output.read_text(encoding="utf-8")
    assert "partial/noncanonical subset" in report
    assert "**Pipeline decision:**" in report
    assert "Baseline probability and calibration" in report
    assert "macro ROC-AUC" in report
    assert "macro AP" in report
    assert "multiclass Brier" in report
    assert "total variation" in report
    assert "KL" in report
    assert "anchor_reliability_bins_full_test" not in report
    assert "all 20,000 test rows" not in report
    assert "full held-out test set" in report
    assert "Reliability panels use" in report
    assert "seen during development" in report
    assert "categorical_probability_validation_results.json" in report
    assert "validation.categorical_probability_validation" in report
    assert "validation.categorical_probability_report" in report
    images = sorted((tmp_path / "figures").glob("*.png"))
    assert images
    assert any("calibration_low_probability" in path.name for path in images)
    assert all(path.stat().st_size > 0 for path in images)
    assert "![" in report


def test_report_loader_accepts_lossless_gzip_json(tmp_path):
    run = _tiny_report_run()
    json_path = write_probability_results(run, tmp_path / "results.json")
    gzip_path = tmp_path / "results.json.gz"
    with json_path.open("rb") as source, gzip.GzipFile(
        filename=str(gzip_path), mode="wb", mtime=0
    ) as target:
        target.write(source.read())
    loaded = load_probability_run(gzip_path)
    assert loaded["protocol"] == run["protocol"]


def test_report_validation_rejects_missing_records():
    run = _tiny_report_run()
    run["records"] = []
    try:
        validate_probability_run(run)
    except ValueError as error:
        assert "at least one fitted record" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected empty runs to be rejected.")
