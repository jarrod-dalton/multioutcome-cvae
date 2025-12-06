from .model import (
    MultivariateOutcomeCVAE,
    CVAETrainer,
    tune_cvae_random_search,
    tune_cvae_tpe,
    fit_cvae_with_tuning,
)

from .simulate import simulate_cvae_data

# Import all diagnostics utilities
from .utils_diagnostics import (
    calibration_curve_with_ci,
    plot_global_calibration,
    plot_per_outcome_calibration_grid,
    expected_calibration_error,
    maximum_calibration_error,
    dependence_curve,
    plot_dependence_curve,
    posterior_predictive_check_gaussian,
    posterior_predictive_check_poisson,
    conditional_ppc_by_feature_decile,
)

__all__ = [
    # Core model + training tools
    "MultivariateOutcomeCVAE",
    "CVAETrainer",
    "tune_cvae_random_search",
    "tune_cvae_tpe",
    "fit_cvae_with_tuning",
    "simulate_cvae_data",

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
]
