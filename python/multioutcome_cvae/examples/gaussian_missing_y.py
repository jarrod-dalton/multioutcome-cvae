"""
examples/gaussian_missing_y.py

Example: training a Gaussian CVAE with partially missing Y.

Steps:
  1. Simulate complete Gaussian data (X, Y_full)
  2. Introduce missingness in Y (MCAR)
  3. Create an imputed version of Y for encoder input
  4. Train CVAE with a Y-mask so the loss only uses observed entries
  5. Compare predictive means to the true Y on observed test entries

Run:

    python -m multioutcome_cvae.examples.gaussian_missing_y
"""

import numpy as np

from multioutcome_cvae import (
    simulate_cvae_data,
    CVAETrainer,
)


def main():
    rng = np.random.default_rng(2025)

    # 1. Simulate complete Gaussian data
    X, Y_full, params = simulate_cvae_data(
        n_samples=5000,
        n_features=5,
        n_outcomes=6,
        latent_dim=2,
        outcome_type="gaussian",
        seed=1234,
        noise_sd=1.0,
    )

    print("Simulated Gaussian data:")
    print("  X shape:", X.shape)
    print("  Y_full shape:", Y_full.shape)

    # 2. Introduce MCAR missingness in Y (30% missing)
    missing_prob = 0.3
    missing_mask = rng.random(Y_full.shape) < missing_prob

    Y_mask = (~missing_mask).astype(np.float32)  # 1 = observed, 0 = missing

    Y_obs = Y_full.copy()
    Y_obs[missing_mask] = np.nan

    # 3. Simple column-mean imputation for encoder input
    col_means = np.nanmean(Y_obs, axis=0)
    Y_imputed = np.where(np.isnan(Y_obs), col_means, Y_obs)

    # 4. Train/validation split
    n = X.shape[0]
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    idx_train, idx_val = idx[:n_train], idx[n_train:]

    X_train, X_val = X[idx_train], X[idx_val]
    Y_train, Y_val = Y_imputed[idx_train], Y_imputed[idx_val]
    Y_mask_train, Y_mask_val = Y_mask[idx_train], Y_mask[idx_val]
    Y_full_val = Y_full[idx_val]  # keep the true values for evaluation

    # 5. Fit Gaussian CVAE with masked Y
    trainer = CVAETrainer(
        x_dim=X.shape[1],
        y_dim=Y_full.shape[1],
        latent_dim=8,
        outcome_type="gaussian",
        hidden_dim=64,
        n_hidden_layers=2,
    )

    history = trainer.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        Y_mask_train=Y_mask_train,
        Y_mask_val=Y_mask_val,
        num_epochs=30,
        verbose=True,
        seed=1234,
    )

    # 6. Predict means on validation set and compute MSE on observed entries
    mu_pred = trainer.predict_mean(X_val, n_mc=30)

    mask_val = Y_mask_val.astype(bool)
    diffs = (mu_pred - Y_full_val) ** 2
    mse_obs = diffs[mask_val].mean()

    print("\nValidation MSE on observed Y entries:", float(mse_obs))


if __name__ == "__main__":
    main()
