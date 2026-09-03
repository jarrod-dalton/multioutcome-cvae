from dataclasses import replace

import numpy as np
import pytest

from validation.categorical_probability_validation import (
    DEFAULT_CONFIG,
    IndependentSoftmaxBaseline,
    aggregate_cell_engineering_status,
    build_probability_protocol_manifest,
    build_probability_scenarios,
    dgp_conditional_probabilities,
    engineering_tolerances,
    evaluate_probability_scenario_seed,
    oracle_expected_probability_metrics,
    population_mean_probabilities,
    probability_protocol_sha256,
    quadrature_convergence_metrics,
    simulate_probability_splits,
    tensor_gauss_legendre_uniform,
)


def _tiny_config(**updates):
    base = replace(
        DEFAULT_CONFIG,
        data_seeds=(101,),
        sample_sizes=(24,),
        validation_sizes=((24, 12),),
        n_test=30,
        focal_prevalences=(0.10,),
        homogeneous_cardinalities=(2,),
        semantic_outcomes=2,
        include_heterogeneity=False,
        include_rho_control=False,
        population_quadrature_order=5,
        num_epochs=2,
        batch_size=12,
        kl_warmup_epochs=1,
        early_stopping_start_epoch=1,
        early_stopping_patience=1,
        baseline_max_iter=20,
        fitted_quadrature_order=3,
        fitted_quadrature_check_order=5,
        quadrature_check_rows=10,
        quadrature_x_batch_size=5,
        calibration_bins=3,
        plot_rows=10,
        include_initialization_sensitivity=False,
    )
    return replace(base, **updates)


def test_default_grid_has_locked_factorial_and_controls():
    scenarios = build_probability_scenarios()
    assert len(scenarios) == 34
    assert sum(item.design == "core" for item in scenarios) == 27
    assert sum(item.design == "heterogeneity" for item in scenarios) == 6
    assert sum(item.design == "rho_control" for item in scenarios) == 1
    core_cells = {
        (
            item.n_train,
            item.focal_prevalence,
            item.cardinalities[0],
        )
        for item in scenarios
        if item.design == "core"
    }
    assert core_cells == {
        (n, q, k)
        for n in (500, 2000, 8000)
        for q in (0.01, 0.05, 0.20)
        for k in (2, 5, 20)
    }
    assert DEFAULT_CONFIG.data_seeds == (5003, 10007, 20011, 40009, 80021)


def test_uniform_tensor_quadrature_integrates_basic_moments():
    nodes, weights = tensor_gauss_legendre_uniform(5, 3)
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights @ nodes, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(weights @ np.square(nodes), np.full(3, 1 / 3), atol=1e-14)


@pytest.mark.parametrize("q", [0.01, 0.05, 0.20])
@pytest.mark.parametrize("k", [2, 5, 20])
def test_focal_prevalence_and_context_balance_are_exact(q, k):
    scenario = next(
        item
        for item in build_probability_scenarios()
        if item.design == "core"
        and item.n_train == 500
        and item.focal_prevalence == q
        and item.cardinalities[0] == k
    )
    means = population_mean_probabilities(
        scenario, order=DEFAULT_CONFIG.population_quadrature_order
    )
    assert means["anchor"][0] == pytest.approx(q, abs=2e-12)
    for outcome in scenario.outcomes:
        if outcome.role == "context":
            np.testing.assert_allclose(
                means[outcome.name],
                np.full(outcome.cardinality, 1.0 / outcome.cardinality),
                atol=2e-12,
            )


def test_splits_are_bounded_disjoint_and_probability_aligned():
    config = _tiny_config()
    scenario = build_probability_scenarios(config)[0]
    first = simulate_probability_splits(scenario, 101, config)
    second = simulate_probability_splits(scenario, 101, config)
    for left, right in zip(
        (first.train, first.validation, first.test),
        (second.train, second.validation, second.test),
    ):
        np.testing.assert_array_equal(left.X, right.X)
        np.testing.assert_array_equal(left.Y, right.Y)
        assert np.all(left.X >= -1.0) and np.all(left.X <= 1.0)
        recalculated = dgp_conditional_probabilities(scenario, left.X)
        for name in recalculated:
            np.testing.assert_allclose(
                left.oracle_probabilities[name], recalculated[name], atol=0.0
            )
    assert not (set(first.train.row_ids) & set(first.validation.row_ids))
    assert not (set(first.train.row_ids) & set(first.test.row_ids))
    assert not (set(first.validation.row_ids) & set(first.test.row_ids))


def test_matched_heterogeneity_pair_has_identical_anchor_data():
    config = replace(
        DEFAULT_CONFIG,
        data_seeds=(101,),
        sample_sizes=(30,),
        validation_sizes=((30, 10),),
        n_test=40,
        focal_prevalences=(0.05,),
        homogeneous_cardinalities=(5,),
        include_rho_control=False,
        include_initialization_sensitivity=False,
    )
    pair = [
        item
        for item in build_probability_scenarios(config)
        if item.design == "heterogeneity"
    ]
    assert len(pair) == 2
    left = simulate_probability_splits(pair[0], 101, config)
    right = simulate_probability_splits(pair[1], 101, config)
    for role_name in ("train", "validation", "test"):
        left_role = getattr(left, role_name)
        right_role = getattr(right, role_name)
        np.testing.assert_array_equal(left_role.X, right_role.X)
        np.testing.assert_array_equal(
            left_role.Y[:, pair[0].anchor_index],
            right_role.Y[:, pair[1].anchor_index],
        )
        np.testing.assert_allclose(
            left_role.oracle_probabilities["anchor"],
            right_role.oracle_probabilities["anchor"],
            atol=0.0,
        )


def test_independent_softmax_is_deterministic_and_normalized():
    rng = np.random.default_rng(33)
    X = rng.uniform(-1.0, 1.0, size=(120, 3))
    logits = 0.2 + X @ np.array([0.8, -0.4, 0.3])
    probability = 1.0 / (1.0 + np.exp(-logits))
    Y = (rng.random(120) < probability).astype(np.int32)[:, None]
    schema = [{"name": "event", "levels": ["no", "yes"]}]
    fits = []
    for _ in range(2):
        model = IndependentSoftmaxBaseline(schema, 3)
        metadata = model.fit(X, Y, l2=1e-3, max_iter=60, tolerance_grad=1e-8)
        predicted = model.predict_probabilities(X)["event"]
        assert metadata["objective"] < np.log(2.0)
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=1e-12)
        fits.append(predicted)
    np.testing.assert_allclose(fits[0], fits[1], atol=0.0)


def test_oracle_metrics_have_exact_brier_tv_kl_and_focal_errors():
    schema = [{"name": "anchor", "levels": ["focal", "other"]}]
    oracle = {"anchor": np.array([[0.2, 0.8], [0.7, 0.3]])}
    fitted = {"anchor": np.array([[0.3, 0.7], [0.6, 0.4]])}
    result = oracle_expected_probability_metrics(
        fitted,
        oracle,
        schema,
        anchor_outcome="anchor",
        focal_class_index=0,
        epsilon=1e-12,
    )
    assert result["focal"]["bias"] == pytest.approx(0.0)
    assert result["focal"]["mae"] == pytest.approx(0.1)
    assert result["focal"]["rmse"] == pytest.approx(0.1)
    assert result["summary"]["mean_outcome_total_variation"] == pytest.approx(0.1)
    assert result["summary"]["mean_outcome_expected_brier_regret"] == pytest.approx(0.02)
    expected_kl = np.mean(
        np.sum(oracle["anchor"] * np.log(oracle["anchor"] / fitted["anchor"]), axis=1)
    )
    assert result["summary"]["mean_outcome_kl_regret"] == pytest.approx(expected_kl)


def test_quadrature_metric_and_tolerances_are_literal():
    schema = [{"name": "a", "levels": ["0", "1"]}]
    primary = {"a": np.array([[0.2, 0.8], [0.5, 0.5]])}
    check = {"a": np.array([[0.21, 0.79], [0.5, 0.5]])}
    result = quadrature_convergence_metrics(primary, check, schema)
    assert result["mean_absolute_probability_change"] == pytest.approx(0.005)
    tolerance = engineering_tolerances(0.01)
    assert tolerance["focal_mae_max"] == pytest.approx(0.005)
    assert tolerance["absolute_focal_bias_max"] == pytest.approx(0.0025)
    assert tolerance["focal_p95_absolute_error_max"] == pytest.approx(0.02)


def test_manifest_is_deterministic_and_contains_coefficients_and_tolerances():
    first = build_probability_protocol_manifest()
    second = build_probability_protocol_manifest()
    assert first == second
    assert probability_protocol_sha256() == probability_protocol_sha256()
    assert first["scenarios"][0]["outcomes"][0]["x_weights"]
    assert "focal_mae" in first["engineering_tolerances"]


def test_tiny_end_to_end_fit_exposes_report_contract():
    config = _tiny_config()
    scenario = build_probability_scenarios(config)[0]
    result = evaluate_probability_scenario_seed(
        scenario,
        101,
        config,
        retain_plot_payload=True,
    )
    assert set(result["model_diagnostics"]["models"]) == {
        "cvae",
        "independent_softmax",
        "intercept_null",
        "oracle",
    }
    assert np.isfinite(
        result["oracle_metrics"]["cvae"]["summary"]["mean_outcome_kl_regret"]
    )
    assert result["plot_payload"]["Y"].shape == (10, 2)
    assert "oracle" not in result["plot_payload"]["model_probabilities"]
    assert result["plot_payload"]["anchor_reliability_bins_full_test"]["cvae"][0]["n"] == 30
    assert "reliability_bins" not in result["model_diagnostics"]["models"]["cvae"]["classes"][0]
    assert result["focal_true_probability_bands"]["cvae"]
    assert result["quadrature_convergence"]["p99_absolute_probability_change"] >= 0
    statuses = aggregate_cell_engineering_status([result])
    assert len(statuses) == 1
    assert statuses[0]["n_replicates"] == 1
