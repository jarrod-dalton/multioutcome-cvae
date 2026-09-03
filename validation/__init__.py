"""Reusable confirmatory validation utilities for the categorical CVAE."""

from importlib import import_module


__all__ = [
    "DEFAULT_CONFIG",
    "PREDECLARED_GATES",
    "SCENARIOS",
    "DataSplit",
    "OutcomeDGP",
    "ScenarioDGP",
    "StrictSplits",
    "ValidationConfig",
    "assert_strict_splits",
    "build_protocol_manifest",
    "collect_environment_metadata",
    "cvae_nested_mc_diagnostic",
    "cvae_quadrature_predictions",
    "dgp_conditional_probabilities",
    "dgp_oracle_log_mass",
    "dgp_oracle_probabilities",
    "evaluate_predeclared_gates",
    "evaluate_scenario_seed",
    "integrate_shared_and_product_log_masses",
    "label_permutation_invariance_check",
    "marginal_predictive_metrics",
    "paired_score_difference",
    "protocol_manifest_sha256",
    "render_markdown_report",
    "run_validation",
    "simulate_strict_splits",
    "summarize_gate_results",
    "tensor_gauss_hermite",
    "write_markdown_report",
]


def __getattr__(name):
    # Lazy exports keep ``python -m validation.categorical_predictive_validation``
    # from importing the executable module twice via package initialization.
    if name in __all__:
        module = import_module(".categorical_predictive_validation", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
