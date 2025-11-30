"""
examples/bernoulli_basic.py

Hello-world example for multivariate Bernoulli outcomes:

  1. Simulate Bernoulli data (X, Y)
  2. Fit a CVAETrainer (outcome_type="bernoulli")
  3. Compute log-likelihood metrics on a test subset
  4. Compare correlation structure (real vs generated)
  5. Optionally display plots

Run:

    python -m multibin_cvae.examples.bernoulli_basic
"""

import numpy as np
import matplotlib.pyplot as plt

from multibin_cvae import (
    simulate_cvae_data,
    CVAETrainer,
    compare_real_vs_generated,
)


def main(show_plots: bool = True):
    # -----------------------
    # 1. Simulate data
    # -----------------------
    X, Y, params_true = simulate_cvae_data(
        n_samples=5000,
        n_features=5,
        n_outcomes=10,
        latent_dim=2,
        outcome_type="bernoulli",
        seed=1234,
    )

    print("Simulated Bernoulli data:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  latent_dim (true): {params_true['latent_dim']}")

    # -----------------------
    # 2. Fit CVAETrainer
    # -----------------------
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y.shape[1],
        latent_dim=8,
        outcome_type="bernoulli",
        hidden_dim=64,
        n_hidden_layers=2,
    )

    history = trainer.fit(X, Y, verbose=True)

    # -----------------------
    # 3. Evaluate on test subset
    # -----------------------
    rng = np.random.default_rng(2025)
    test_idx = rng.choice(X.shape[0], size=1000, replace=False)
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    metrics = trainer.evaluate_loglik(X_test, Y_test, n_mc=30)
    print("\nLog-likelihood metrics on test subset:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # -----------------------
    # 4. Correlation: real vs generated
    # -----------------------
    # Generate one sample Y per test X and flatten
    Y_gen = trainer.generate(X_test, n_samples_per_x=1, return_probs=False)
    Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

    comp = compare_real_vs_generated(
        Y_real=Y_test,
        Y_gen=Y_gen_flat,
        label_real="test",
        label_gen="generated",
        make_plot=show_plots,
    )

    print("\nCorrelation summary:")
    print("  corr_real[0:3, 0:3]:")
    print(comp["corr_real"][:3, :3])
    print("  corr_gen[0:3, 0:3]:")
    print(comp["corr_gen"][:3, :3])

    if show_plots and comp["fig"] is not None:
        plt.show()


if __name__ == "__main__":
    main()
