from .model import (
    MultivariateOutcomeCVAE,
    CVAETrainer,
    tune_cvae_random_search,
    tune_cvae_tpe,
    fit_cvae_with_tuning,
)

from .simulate import (
    compare_real_vs_generated,
    simulate_cvae_data,
    summarize_binary_matrix,
)
from .export_r import export_bernoulli_r

_DIAGNOSTIC_EXPORTS = {
    "calibration_curve_with_ci",
    "plot_global_calibration",
    "plot_per_outcome_calibration_grid",
    "expected_calibration_error",
    "maximum_calibration_error",
    "dependence_curve",
    "plot_dependence_curve",
    "posterior_predictive_check_gaussian",
    "posterior_predictive_check_poisson",
    "conditional_ppc_by_feature_decile",
    "bernoulli_dependence_metrics",
}


def __getattr__(name):
    if name in _DIAGNOSTIC_EXPORTS:
        from . import utils_diagnostics

        value = getattr(utils_diagnostics, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core model + training tools
    "MultivariateOutcomeCVAE",
    "CVAETrainer",
    "tune_cvae_random_search",
    "tune_cvae_tpe",
    "fit_cvae_with_tuning",
    "simulate_cvae_data",
    "summarize_binary_matrix",
    "compare_real_vs_generated",
    "export_bernoulli_r",

    # Diagnostics tools
    "calibration_curve_with_ci",
    "plot_global_calibration",
    "plot_per_outcome_calibration_grid",
    "expected_calibration_error",
    "maximum_calibration_error",
    "dependence_curve",
    "plot_dependence_curve",
    "posterior_predictive_check_gaussian",
    "posterior_predictive_check_poisson",
    "conditional_ppc_by_feature_decile",
    "bernoulli_dependence_metrics",
]
