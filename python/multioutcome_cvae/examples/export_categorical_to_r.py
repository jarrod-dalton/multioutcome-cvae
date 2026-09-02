"""Fit a neutral categorical CVAE and export dependency-free base-R source."""

from pathlib import Path

import torch

from multioutcome_cvae import CVAETrainer, export_categorical_r
from multioutcome_cvae.examples.categorical_basic import (
    OUTCOME_SCHEMA,
    simulate_mixed_categorical_data,
)


def main(output_path: str = "categorical_cvae.R") -> Path:
    X, Y = simulate_mixed_categorical_data(n_samples=5000, seed=2026)
    torch.manual_seed(2026)
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=4,
        outcome_type="categorical",
        outcome_schema=OUTCOME_SCHEMA,
        hidden_dim=64,
        n_hidden_layers=2,
    )
    trainer.fit(
        X,
        Y,
        num_epochs=30,
        kl_warmup_epochs=10,
        seed=2026,
        verbose=True,
    )

    path = export_categorical_r(
        trainer,
        output_path,
        feature_names=[f"x{index + 1}" for index in range(X.shape[1])],
    )
    print(f"Exported standalone categorical R model to {path.resolve()}")
    return path


if __name__ == "__main__":
    main()
