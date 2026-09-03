"""Fast deterministic checks for the categorical predictive-validation protocol."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest

from validation import (
    DEFAULT_CONFIG,
    PREDECLARED_GATES,
    SCENARIOS,
    DataSplit,
    StrictSplits,
    assert_strict_splits,
    dgp_conditional_probabilities,
    dgp_oracle_log_mass,
    dgp_oracle_probabilities,
    evaluate_predeclared_gates,
    evaluate_scenario_seed,
    integrate_shared_and_product_log_masses,
    label_permutation_invariance_check,
    marginal_predictive_metrics,
    paired_score_difference,
    render_markdown_report,
    simulate_strict_splits,
    tensor_gauss_hermite,
)
from validation.categorical_predictive_validation import summarize_gate_results


def _small_config(**changes):
    settings = {
        "seeds": (101, 202),
        "n_train": 17,
        "n_validation": 7,
        "n_test": 9,
    }
    settings.update(changes)
    return replace(DEFAULT_CONFIG, **settings)


def _valid_codes(scenario, n_rows):
    rows = np.arange(n_rows)
    return np.column_stack(
        [rows % len(outcome.levels) for outcome in scenario.outcomes]
    ).astype(np.int32)


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_strict_split_simulation_is_reproducible_disjoint_and_schema_valid(scenario):
    config = _small_config()
    first = simulate_strict_splits(scenario, seed=8128, config=config)
    second = simulate_strict_splits(scenario, seed=8128, config=config)

    assert_strict_splits(first, config.n_train + config.n_validation + config.n_test)
    names = [outcome.name for outcome in scenario.outcomes]
    assert names == [entry["name"] for entry in scenario.outcome_schema]
    assert len(names) == len(set(names))

    for first_role, second_role, expected_rows in zip(
        (first.train, first.validation, first.test),
        (second.train, second.validation, second.test),
        (config.n_train, config.n_validation, config.n_test),
    ):
        np.testing.assert_array_equal(first_role.X, second_role.X)
        np.testing.assert_array_equal(first_role.Y, second_role.Y)
        np.testing.assert_array_equal(first_role.row_ids, second_role.row_ids)
        assert first_role.X.shape == (expected_rows, scenario.x_dim)
        assert first_role.Y.shape == (expected_rows, len(scenario.outcomes))
        assert first_role.X.dtype == np.float32
        assert first_role.Y.dtype == np.int32
        assert np.isfinite(first_role.X).all()
        for column, outcome in enumerate(scenario.outcomes):
            assert len(outcome.levels) >= 2
            assert len(outcome.levels) == len(set(outcome.levels))
            assert np.all(first_role.Y[:, column] >= 0)
            assert np.all(first_role.Y[:, column] < len(outcome.levels))


def test_split_roles_use_independent_deterministic_rng_streams():
    scenario = SCENARIOS["mixed_dependent"]
    original = simulate_strict_splits(scenario, 77, _small_config())
    resized_training = simulate_strict_splits(
        scenario,
        77,
        _small_config(n_train=23),
    )

    # Changing the amount of training data cannot move the validation or test
    # draws because each role is generated from its own SeedSequence child.
    np.testing.assert_array_equal(original.validation.X, resized_training.validation.X)
    np.testing.assert_array_equal(original.validation.Y, resized_training.validation.Y)
    np.testing.assert_array_equal(original.test.X, resized_training.test.X)
    np.testing.assert_array_equal(original.test.Y, resized_training.test.Y)

    different_seed = simulate_strict_splits(scenario, 78, _small_config())
    assert not np.array_equal(original.train.X, different_seed.train.X)


def test_strict_split_assertion_rejects_role_overlap():
    scenario = SCENARIOS["mixed_dependent"]
    splits = simulate_strict_splits(scenario, 31, _small_config())
    overlapping_validation = DataSplit(
        X=splits.validation.X,
        Y=splits.validation.Y,
        row_ids=splits.validation.row_ids.copy(),
    )
    overlapping_validation.row_ids[0] = splits.train.row_ids[0]
    tampered = StrictSplits(splits.train, overlapping_validation, splits.test)

    with pytest.raises(RuntimeError, match="overlap"):
        assert_strict_splits(
            tampered,
            _small_config().n_train
            + _small_config().n_validation
            + _small_config().n_test,
        )


@pytest.mark.parametrize(
    "scenario_name",
    ("mixed_conditional_independence", "all_binary_conditional_independence"),
)
def test_conditional_independence_dgps_ignore_z_and_oracle_factorizes(scenario_name):
    scenario = SCENARIOS[scenario_name]
    X = np.array(
        [[-1.2, 0.1, 2.0], [0.0, 0.0, 0.0], [0.8, -0.3, 0.1]],
        dtype=np.float64,
    )
    probabilities_at_zero = dgp_conditional_probabilities(
        scenario, X, np.zeros((X.shape[0], scenario.latent_dim))
    )
    probabilities_at_other_z = dgp_conditional_probabilities(
        scenario,
        X,
        np.array([[20.0, -13.0], [-4.0, 7.0], [9.0, 2.5]]),
    )
    oracle_probabilities = dgp_oracle_probabilities(scenario, X, order=9)

    for outcome in scenario.outcomes:
        name = outcome.name
        np.testing.assert_allclose(
            probabilities_at_zero[name], probabilities_at_other_z[name], atol=0.0
        )
        np.testing.assert_allclose(
            oracle_probabilities[name], probabilities_at_zero[name], atol=2e-15
        )

    oracle_mass = dgp_oracle_log_mass(
        scenario, X, _valid_codes(scenario, X.shape[0]), order=9
    )
    np.testing.assert_allclose(
        oracle_mass["joint"],
        oracle_mass["fitted_marginal_product"],
        atol=2e-15,
    )


def test_shared_and_product_integrator_matches_direct_probability_arithmetic():
    weights = np.array([0.25, 0.75])
    first = np.array([[0.2, 0.5], [0.6, 0.1]])
    second = np.array([[0.3, 0.4], [0.7, 0.9]])
    integrated = integrate_shared_and_product_log_masses(
        [np.log(first), np.log(second)], weights
    )

    expected_shared = np.log((weights[:, None] * first * second).sum(axis=0))
    expected_product = np.log(
        (weights[:, None] * first).sum(axis=0)
        * (weights[:, None] * second).sum(axis=0)
    )
    np.testing.assert_allclose(integrated["shared"], expected_shared)
    np.testing.assert_allclose(integrated["product"], expected_product)
    np.testing.assert_allclose(
        integrated["gain"], expected_shared - expected_product
    )

    # Log-space accumulation must remain finite well below exp() underflow.
    extreme = integrate_shared_and_product_log_masses(
        [
            np.array([[-10_000.0], [-10_001.0]]),
            np.array([[-20_000.0], [-20_002.0]]),
        ],
        weights,
    )
    assert all(np.isfinite(values).all() for values in extreme.values())


def test_gauss_hermite_grid_is_normalized_and_has_standard_normal_moments():
    nodes, weights = tensor_gauss_hermite(order=5, dimensions=2)

    assert nodes.shape == (25, 2)
    assert weights.shape == (25,)
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=2e-15)
    np.testing.assert_allclose(weights @ nodes, np.zeros(2), atol=2e-15)
    np.testing.assert_allclose(weights @ np.square(nodes), np.ones(2), atol=2e-15)
    np.testing.assert_allclose(weights @ (nodes[:, 0] * nodes[:, 1]), 0.0, atol=2e-15)
    np.testing.assert_allclose(weights @ np.power(nodes, 4), np.full(2, 3.0), atol=2e-14)


@pytest.mark.parametrize(
    "scenario_name", ("mixed_dependent", "all_binary_dependent")
)
def test_dgp_oracle_quadrature_converges_for_dependent_scenarios(scenario_name):
    scenario = SCENARIOS[scenario_name]
    X = np.array(
        [[-1.2, 0.1, 2.0], [0.0, 0.0, 0.0], [0.8, -0.3, 0.1]],
        dtype=np.float64,
    )
    order_9 = dgp_oracle_probabilities(scenario, X, order=9)
    order_25 = dgp_oracle_probabilities(scenario, X, order=25)

    for outcome in scenario.outcomes:
        np.testing.assert_allclose(
            order_9[outcome.name], order_25[outcome.name], atol=7e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            order_25[outcome.name].sum(axis=1), np.ones(X.shape[0]), atol=2e-15
        )


def test_marginal_proper_scores_match_hand_calculation():
    schema = [
        {"name": "binary", "levels": ["no", "yes"]},
        {"name": "three", "levels": ["a", "b", "c"]},
    ]
    Y = np.array([[0, 2], [1, 1]], dtype=np.int32)
    probabilities = {
        "binary": np.array([[0.8, 0.2], [0.25, 0.75]]),
        "three": np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3]]),
    }
    metrics = marginal_predictive_metrics(probabilities, Y, schema, n_bins=2)

    expected_row_nll = -np.array(
        [(np.log(0.8) + np.log(0.7)) / 2, (np.log(0.75) + np.log(0.5)) / 2]
    )
    expected_row_brier = np.array([(0.08 + 0.14) / 2, (0.125 + 0.38) / 2])
    np.testing.assert_allclose(metrics["per_row_nll"], expected_row_nll)
    np.testing.assert_allclose(metrics["per_row_brier"], expected_row_brier)
    assert metrics["nll"] == pytest.approx(expected_row_nll.mean())
    assert metrics["brier"] == pytest.approx(expected_row_brier.mean())

    paired = paired_score_difference(
        candidate=np.array([3.0, 5.0, 7.0]),
        reference=np.array([2.0, 4.0, 6.0]),
    )
    assert paired["mean"] == pytest.approx(1.0)
    assert paired["standard_error"] == pytest.approx(0.0)
    assert paired["lcb_95_one_sided"] == pytest.approx(1.0)


def test_all_reported_scores_are_invariant_to_lossless_label_permutation():
    schema = [
        {"name": "binary", "levels": ["no", "yes"]},
        {"name": "three", "levels": ["a", "b", "c"]},
    ]
    Y = np.array([[0, 2], [1, 1], [1, 0], [0, 1]], dtype=np.int32)
    probabilities = {
        "binary": np.array(
            [[0.8, 0.2], [0.25, 0.75], [0.1, 0.9], [0.65, 0.35]]
        ),
        "three": np.array(
            [
                [0.1, 0.2, 0.7],
                [0.2, 0.5, 0.3],
                [0.6, 0.3, 0.1],
                [0.2, 0.6, 0.2],
            ]
        ),
    }
    check = label_permutation_invariance_check(
        probabilities,
        Y,
        schema,
        permutations=((1, 0), (2, 0, 1)),
        n_bins=3,
    )

    assert check["permutations_new_to_old"] == [[1, 0], [2, 0, 1]]
    assert check["maximum_absolute_discrepancy"] <= 1.0e-15
    assert all(
        change <= 1.0e-15
        for change in check["metric_absolute_changes"].values()
    )

    with pytest.raises(ValueError, match="Invalid label permutation"):
        label_permutation_invariance_check(
            probabilities,
            Y,
            schema,
            permutations=((0, 0), (0, 1, 2)),
        )


def _favorable_gate_inputs():
    results = []
    for scenario in SCENARIOS.values():
        for seed in (101, 202, 303):
            dependent = not scenario.conditionally_independent
            results.append(
                {
                    "scenario": scenario.name,
                    "seed": seed,
                    "conditionally_independent": scenario.conditionally_independent,
                    "cvae_joint_gain": {"mean": 0.03 if dependent else 0.0},
                    # Equality is intentional: this gate is declared >=.
                    "oracle_joint_gain": {"mean": 0.05 if dependent else 0.0},
                    "marginal_log_score_vs_intercept": {"mean": 0.10},
                    "marginal_brier_improvement_vs_intercept": {"mean": 0.05},
                    "quadrature_convergence": [
                        {
                            "order": 41,
                            # Equalities are intentional: these gates are <=.
                            "gain_mean_absolute_change": 0.002,
                            "score_or_gain_absolute_change_p99": 0.01,
                        }
                    ],
                    "label_permutation_checks": {
                        model: {"maximum_absolute_discrepancy": 1.0e-10}
                        for model in ("cvae", "intercept_null", "oracle")
                    },
                }
            )
    return results


def test_predeclared_gates_apply_documented_strict_and_inclusive_boundaries():
    gates = evaluate_predeclared_gates(_favorable_gate_inputs())

    assert {gate["gate"] for gate in gates} == set(PREDECLARED_GATES)
    assert all(gate["passed"] for gate in gates)
    assert {gate["operator"] for gate in gates} == {">", "<", ">=", "<="}


def test_predeclared_gate_failures_remain_visible_without_aborting_evaluation():
    results = copy.deepcopy(_favorable_gate_inputs())
    for result in results:
        if result["scenario"] == "mixed_dependent" and result["seed"] == 101:
            result["cvae_joint_gain"]["mean"] = -0.001
        if result["scenario"] == "mixed_conditional_independence":
            # Equality fails because the per-seed CI penalty gate is strict <.
            result["cvae_joint_gain"]["mean"] = -0.02
        if result["scenario"] == "all_binary_conditional_independence":
            # Equality fails because the marginal-improvement gate is strict >.
            result["marginal_log_score_vs_intercept"]["mean"] = 0.0

    gates = evaluate_predeclared_gates(results)
    indexed = {(gate["gate"], gate["scope"]): gate for gate in gates}

    assert not indexed[("dependent_every_seed_positive", "mixed_dependent")][
        "passed"
    ]
    assert not indexed[
        ("conditional_independence_each_seed", "mixed_conditional_independence")
    ]["passed"]
    assert not indexed[
        ("marginal_log_score_lcb", "all_binary_conditional_independence")
    ]["passed"]
    assert any(not gate["passed"] for gate in gates)
    assert any(gate["passed"] for gate in gates)


def test_aggregate_joint_gates_cluster_scenarios_within_top_level_seed():
    results = _favorable_gate_inputs()
    dependent_gains = {
        "mixed_dependent": {101: 0.01, 202: 0.02, 303: 0.03},
        "all_binary_dependent": {101: 0.09, 202: 0.18, 303: 0.27},
    }
    ci_gains = {
        "mixed_conditional_independence": {101: -0.01, 202: -0.02, 303: -0.03},
        "all_binary_conditional_independence": {
            101: -0.09,
            202: -0.18,
            303: -0.27,
        },
    }
    for result in results:
        scenario = result["scenario"]
        seed = result["seed"]
        if scenario in dependent_gains:
            result["cvae_joint_gain"]["mean"] = dependent_gains[scenario][seed]
        elif scenario in ci_gains:
            result["cvae_joint_gain"]["mean"] = ci_gains[scenario][seed]

    gates = evaluate_predeclared_gates(results)
    indexed = {gate["gate"]: gate for gate in gates}
    clustered = np.array([0.05, 0.10, 0.15])
    critical_value = 2.919985580  # one-sided 95% Student-t, df=2
    standard_error = clustered.std(ddof=1) / np.sqrt(clustered.size)
    expected_lcb = clustered.mean() - critical_value * standard_error
    expected_ucb = clustered.mean() + critical_value * standard_error

    assert indexed["dependent_aggregate_lcb"]["value"] == pytest.approx(
        expected_lcb
    )
    assert indexed["conditional_independence_penalty_ucb"][
        "value"
    ] == pytest.approx(expected_ucb)

    # Treating the two same-seed scenarios as six independent replications
    # gives different bounds and would be pseudoreplication.
    unclustered = np.array([0.01, 0.02, 0.03, 0.09, 0.18, 0.27])
    unclustered_se = unclustered.std(ddof=1) / np.sqrt(unclustered.size)
    assert indexed["dependent_aggregate_lcb"]["value"] != pytest.approx(
        unclustered.mean() - 2.015048373 * unclustered_se
    )
    assert indexed["conditional_independence_penalty_ucb"][
        "value"
    ] != pytest.approx(unclustered.mean() + 2.015048373 * unclustered_se)


@pytest.mark.parametrize(
    ("predictive_passed", "prerequisite_passed", "expected"),
    (
        (False, False, "inconclusive"),
        (False, True, "fail"),
        (True, True, "pass"),
    ),
)
def test_gate_disposition_prioritizes_prerequisites(
    predictive_passed, prerequisite_passed, expected
):
    summary = summarize_gate_results(
        [
            {"kind": "predictive", "passed": predictive_passed},
            {"kind": "prerequisite", "passed": prerequisite_passed},
        ]
    )

    assert summary["disposition"] == expected
    assert summary["predictive"] == {
        "passed": int(predictive_passed),
        "failed": int(not predictive_passed),
        "total": 1,
    }
    assert summary["prerequisite"] == {
        "passed": int(prerequisite_passed),
        "failed": int(not prerequisite_passed),
        "total": 1,
    }


def test_markdown_report_renders_disposition_and_separate_gate_counts():
    gate_summary = {
        "disposition": "inconclusive",
        "display": "INCONCLUSIVE",
        "reason": "Synthetic prerequisite failure for renderer coverage.",
        "predictive": {"passed": 2, "failed": 1, "total": 3},
        "prerequisite": {"passed": 4, "failed": 1, "total": 5},
        "all": {"passed": 6, "failed": 2, "total": 8},
    }
    assert not {"passed", "failed", "total"}.intersection(gate_summary)
    environment = {
        "package_version": "test",
        "package_path": "/test/package",
        "trainer_source_path": "/test/model.py",
        "trainer_source_sha256": "trainer-hash",
        "validation_source_path": "/test/validation.py",
        "validation_source_sha256": "validation-hash",
        "package_source_matches_repository": "true",
        "git_head": "test-head",
        "git_dirty": "false",
        "python": "test-python",
        "numpy": "test-numpy",
        "torch": "test-torch",
        "platform": "test-platform",
    }
    report = render_markdown_report(
        {
            "protocol": "synthetic-render-test",
            "device": "cpu",
            "environment": environment,
            "protocol_manifest_sha256": "manifest-hash",
            "protocol_manifest": {},
            "config": asdict(DEFAULT_CONFIG),
            "scenarios": [],
            "gates": [],
            "gate_summary": gate_summary,
            "runtime_seconds": 0.0,
        }
    )

    assert "Phase A disposition: **INCONCLUSIVE**." in report
    assert "Predictive gates: **2 passed / 1 failed / 3 total**." in report
    assert "Prerequisites: **4 passed / 1 failed / 5 total**." in report


def test_tiny_fitted_categorical_validation_path_is_finite_and_structured():
    config = replace(
        DEFAULT_CONFIG,
        seeds=(101, 202, 303),
        n_train=24,
        n_validation=8,
        n_test=10,
        hidden_dim=8,
        n_hidden_layers=1,
        num_epochs=2,
        batch_size=8,
        kl_warmup_epochs=1,
        early_stopping_patience=1,
        early_stopping_start_epoch=1,
        fitted_quadrature_order=5,
        fitted_quadrature_checks=(3, 7),
        oracle_quadrature_order=7,
        oracle_x_batch_size=5,
        quadrature_test_rows=6,
        quadrature_x_batch_size=5,
        mc_sizes=(2, 4),
        mc_test_rows=4,
        mc_blocks=2,
        calibration_bins=2,
    )

    result = evaluate_scenario_seed(
        SCENARIOS["mixed_dependent"],
        seed=config.seeds[0],
        config=config,
        device="cpu",
        verbose=False,
    )

    assert result["scenario"] == "mixed_dependent"
    assert result["seed"] == config.seeds[0]
    assert result["split_integrity"] is True
    assert result["split_sizes"] == {"train": 24, "validation": 8, "test": 10}
    assert 1 <= result["training"]["best_epoch"] <= config.num_epochs
    assert 1 <= result["training"]["epochs_ran"] <= config.num_epochs
    assert np.isfinite(result["training"]["final_train_loss"])
    assert np.isfinite(result["joint"]["cvae"]["nll"])
    assert np.isfinite(result["cvae_joint_gain"]["mean"])
    assert np.isfinite(result["oracle_joint_gain"]["mean"])
    assert {row["order"] for row in result["quadrature_convergence"]} == {3, 7}
    assert all(
        np.isfinite(row["score_or_gain_absolute_change_p99"])
        for row in result["quadrature_convergence"]
    )
    assert [row["draws"] for row in result["mc_diagnostic"]["path"]] == [2, 4]
    assert all(
        np.isfinite(row["mean_gain"])
        for row in result["mc_diagnostic"]["path"]
    )
    assert result["label_permutation_checks"]["cvae"][
        "maximum_absolute_discrepancy"
    ] <= 1.0e-12
