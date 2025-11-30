"""
examples/gaussian_basic.py

Hello-world example for multivariate Gaussian outcomes:

  1. Simulate Gaussian data (X, Y)
  2. Fit a CVAETrainer (outcome_type="gaussian")
  3. Inspect predictive mean / SD via predict_params()
  4. Generate new samples and examine correlation structure

Run:

    python -m multibin_cvae.examples.gaussian_basic
"""

import numpy as np
import matplotlib.pyplot as plt

from multibin_cvae import (
    simulate_cvae_data,
    CVAETrainer,
    summarize_binary_matrix,   # works for numeric too; see docstring
)


def main(show_plots: bool = True):
    # -----------------------
    # 1. Simulate Gaussian data
    # -----------------------
    X, Y, params_true = simulate_cvae_data(
        n_samples=5000,
        n_features=5,
        n_outcomes=6,
        latent_dim=2,
        outcome_type="gaussian",
        seed=5678,
        noise_sd=1.0,
    )

    print("Simulated Gaussian data:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  latent_dim (true): {params_true['latent_dim']}")
    print(f"  noise_sd (true): {params_true['noise_sd']}")

    # -----------------------
    # 2. Fit CVAETrainer
    # -----------------------
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=8,
        outcome_type="gaussian",
        hidden_dim=64,
        n_hidden_layers=2,
    )

    trainer.fit(X, Y, verbose=True)

    # -----------------------
    # 3. Predict distribution parameters
    # -----------------------
    X_sub = X[:5]
    params_pred = trainer.predict_params(X_sub, n_mc=30)

    mu_pred = params_pred["mu"]
    sigma_pred = params_pred["sigma"]

    print("\nPredictive parameters for first 5 observations:")
    print("  mu_pred[0]:", mu_pred[0])
    print("  sigma_pred[0]:", sigma_pred[0])

    # -----------------------
    # 4. Generate samples and inspect correlation
    # -----------------------
    Y_gen = trainer.generate(
        X,
        n_samples_per_x=1,
        return_probs=False
    )
    Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

    summ_real = summarize_binary_matrix(Y, name="gaussian_real", make_plot=False)
    summ_gen = summarize_binary_matrix(Y_gen_flat, name="gaussian_generated", make_plot=False)

    print("\nCorrelation (real) [0:3,0:3]:")
    print(summ_real["corr"][:3, :3])
    print("Correlation (generated) [0:3,0:3]:")
    print(summ_gen["corr"][:3, :3])

    if show_plots:
        # Simple side-by-side heatmaps of corr matrices
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        def _plot_corr(ax, mat, title):
            mat_masked = mat.copy()
            np.fill_diagonal(mat_masked, np.nan)
            im = ax.imshow(mat_masked, aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("Outcome index")
            ax.set_ylabel("Outcome index")
            return im

        im0 = _plot_corr(axes[0], summ_real["corr"], "Real corr")
        im1 = _plot_corr(axes[1], summ_gen["corr"], "Generated corr")
        fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
