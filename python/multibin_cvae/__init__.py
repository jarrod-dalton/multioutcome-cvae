"""
Public API for the multibin_cvae package.

Most users will interact with:

- simulate_cvae_data(...)
- CVAETrainer
- fit_cvae_with_tuning(...)
"""

from .model import (
    MultivariateOutcomeCVAE,
    CVAETrainer,
    tune_cvae_random_search,
    tune_cvae_tpe,
    fit_cvae_with_tuning,
)
from .simulate import simulate_cvae_data

__all__ = [
    "MultivariateOutcomeCVAE",
    "CVAETrainer",
    "tune_cvae_random_search",
    "tune_cvae_tpe",
    "fit_cvae_with_tuning",
    "simulate_cvae_data",
]
