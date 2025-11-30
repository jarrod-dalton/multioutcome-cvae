"""
examples/poisson_basic.py

Hello-world example for multivariate Poisson outcomes:

  1. Simulate Poisson data (X, Y)
  2. Fit a CVAETrainer (outcome_type="poisson")
  3. Inspect predictive rate / Var(Y|X) via predict_params()
  4. Generate new samples and examine correlation structure

Run:

    python -m multibin_cvae.examples.poisson_basic
"""

import numpy as np
import matplotlib.pyplot as plt

from multibin_cvae import (
    simulate_cvae_data,
    CVAETrainer,
    summarize_binary_matrix,   # again, works for numeric matrices
)


def main(show_plots: bool = True):
    # -----------------------
    # 1. Simulate Poisson data
    # -----------------------
    X, Y, params_true = simulate_cvae_data(
        n_samples=5000,
        n_features=5,
        n_outcomes=6,
        latent_dim=2,
        outcome_type="poisson",
        seed=9012,
        base_rate=0.5,
        rate_scale=0.5,
    )

    print("Simulated Poisson data:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  latent_dim (true): {params_true['latent_dim']}")
    print(f"  base_rate (true): {params_true['base_rate']}")
    print(f"  rate_scale (true): {params_true['rate_scale']}")

    # -----------------------
    # 2. Fit CVAETrainer
    # -----------------------
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=8,
        outcome_type="poisson",
        hidden_dim=64,
        n_hidden_layers=2,
    )

    trainer.fit(X, Y, verbose=True)

    # -----------------------
    # 3. Predict distribution parameters
    # -----------------------
    X_sub = X[:5]
    params_pred = trainer.predict_params(X_sub, n_mc=30)

    rate_pred = params_pred["rate"]
    var_y_pred = params_pred["var_y"]

    print("\nPredictive parameters for first 5 observations:")
    print("  rate_pred[0]:", rate_pred[0])
    print("  Var(Y | X)[0]:", var_y_pred[0])

    # -----------------------
    # 4. Generate samples and inspect correlation
    # -----------------------
    Y_gen = trainer.generate(
        X,
        n_samples_per_x=1,
        return_probs=False
    )
    Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

    summ_real = summarize_binary_matrix(Y, name="poisson_real", make_plot=False)
    summ_gen = summarize_binary_matrix(Y_gen_flat, name="poisson_generated", make_plot=False)

    print("\nCorrelation (real) [0:3,0:3]:")
    print(summ_real["corr"][:3, :3])
    print("Correlation (generated) [0:3,0:3]:")
    print(summ_gen["corr"][:3, :3])

    if show_plots:
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
