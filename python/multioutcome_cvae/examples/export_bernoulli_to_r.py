"""Fit a Bernoulli CVAE and export dependency-free base R inference source."""

from pathlib import Path

from multioutcome_cvae import CVAETrainer, export_bernoulli_r, simulate_cvae_data


def main(output_path: str = "cvae_model.R") -> Path:
    X, Y, _ = simulate_cvae_data(
        n_samples=5000,
        n_features=5,
        n_outcomes=10,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=1234,
    )
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=4,
        outcome_type="bernoulli",
        hidden_dim=64,
        n_hidden_layers=2,
    )
    trainer.fit(X, Y, num_epochs=20, seed=1234)

    path = export_bernoulli_r(
        trainer,
        output_path,
        feature_names=[f"x{index + 1}" for index in range(X.shape[1])],
        outcome_names=[f"y{index + 1}" for index in range(Y.shape[1])],
    )
    print(f"Exported standalone R model to {path.resolve()}")
    return path


if __name__ == "__main__":
    main()