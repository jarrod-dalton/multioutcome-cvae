
from .model import (
    XYDataset,
    MultivariateBinaryCVAE,
    CVAETrainer,
    tune_cvae_random_search,
)

from .simulate import (
    simulate_cvae_data,
    train_val_test_split,
    summarize_binary_matrix,
    compare_real_vs_generated,
)

__all__ = [
    "CVAETrainer",
    "simulate_cvae_data",
    "simulate_bernoulli_data",
    "simulate_gaussian_data",
    "simulate_poisson_data",
    "summarize_binary_matrix",
    "compare_real_vs_generated",
    "fit_cvae_with_tuning",
]

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("multioutcome-cvae")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"
