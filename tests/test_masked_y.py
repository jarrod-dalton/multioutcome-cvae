# tests/test_masked_y.py
import numpy as np
import torch

from multioutcome_cvae import CVAETrainer


def test_recon_loss_gaussian_mask_vs_manual():
    """
    For Gaussian outcomes, _recon_loss with a mask should equal the
    analytically computed masked NLL (up to the usual constant).
    """
    y_np = np.array([[1.0, -2.0, 0.5],
                     [0.0,  3.0, -1.5]], dtype=np.float32)
    mask_np = np.array([[1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0]], dtype=np.float32)  # some entries missing

    y = torch.from_numpy(y_np)
    mask = torch.from_numpy(mask_np)

    # Simple decoder output: mu = 0, logvar = 0 → var = 1
    out = {
        "mu": torch.zeros_like(y),
        "logvar": torch.zeros_like(y),
    }

    trainer = CVAETrainer(
        x_dim=4,
        y_dim=3,
        latent_dim=2,
        outcome_type="gaussian",
        hidden_dim=8,
        n_hidden_layers=1,
    )

    loss = trainer._recon_loss(y, out, mask=mask)

    # Manual masked NLL (up to constant):
    # 0.5 * (logvar + (y - mu)^2 / var) with logvar=0, var=1 → 0.5 * y^2
    manual = 0.5 * (y_np ** 2) * mask_np
    manual_sum = manual.sum()

    assert torch.allclose(loss, torch.tensor(manual_sum, dtype=loss.dtype), atol=1e-6)


def test_recon_loss_bernoulli_mask_scaling():
    """
    For Bernoulli outcomes with logits=0 (p=0.5), the per-entry BCE is constant.
    Masking out some entries should reduce the total loss exactly in proportion
    to the number of observed entries.
    """
    y_np = np.array([[0.0, 1.0, 1.0, 0.0],
                     [1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    mask_np = np.array([[1.0, 0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 1.0]], dtype=np.float32)

    y = torch.from_numpy(y_np)
    mask = torch.from_numpy(mask_np)

    # logits = 0 → p = 0.5
    logits = torch.zeros_like(y)
    out = {"logits": logits}

    trainer = CVAETrainer(
        x_dim=3,
        y_dim=4,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=8,
        n_hidden_layers=1,
    )

    # Unmasked loss
    loss_unmasked = trainer._recon_loss(y, out, mask=None)

    # Masked loss
    loss_masked = trainer._recon_loss(y, out, mask=mask)

    # With p = 0.5, BCE per entry is -log(0.5) for any y in {0,1},
    # so the unmasked loss = n_entries * const,
    # and masked loss = n_observed * const.
    n_entries = y_np.size
    n_obs = mask_np.sum()

    ratio = loss_masked.item() / loss_unmasked.item()
    expected_ratio = n_obs / n_entries

    assert abs(ratio - expected_ratio) < 1e-6


def test_fit_with_missing_y_gaussian_runs():
    """
    Smoke test: fitting a small Gaussian CVAE with missing Y
    (masked loss) should run without errors and produce finite losses.
    """
    rng = np.random.default_rng(123)

    n = 200
    x_dim = 3
    y_dim = 4

    X = rng.normal(size=(n, x_dim)).astype(np.float32)
    Y_full = rng.normal(size=(n, y_dim)).astype(np.float32)

    # 30% missing completely at random
    missing_prob = 0.3
    missing = rng.random(size=Y_full.shape) < missing_prob
    Y_mask = (~missing).astype(np.float32)

    Y_obs = Y_full.copy()
    Y_obs[missing] = np.nan

    # Simple column-mean imputation for encoder input
    col_means = np.nanmean(Y_obs, axis=0)
    Y_imputed = np.where(np.isnan(Y_obs), col_means, Y_obs)

    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    idx_train, idx_val = idx[:n_train], idx[n_train:]

    X_train, X_val = X[idx_train], X[idx_val]
    Y_train, Y_val = Y_imputed[idx_train], Y_imputed[idx_val]
    Y_mask_train, Y_mask_val = Y_mask[idx_train], Y_mask[idx_val]

    trainer = CVAETrainer(
        x_dim=x_dim,
        y_dim=y_dim,
        latent_dim=2,
        outcome_type="gaussian",
        hidden_dim=16,
        n_hidden_layers=1,
        num_epochs=5,
        batch_size=32,
    )

    history = trainer.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        Y_mask_train=Y_mask_train,
        Y_mask_val=Y_mask_val,
        num_epochs=5,
        verbose=False,
        seed=42,
    )

    # Basic sanity checks
    assert len(history["train_loss"]) == 5
    assert all(np.isfinite(history["train_loss"]))
    if len(history["val_loss"]) > 0:
        assert all(np.isfinite(history["val_loss"]))
