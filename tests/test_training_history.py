import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer


def test_training_history_reports_weighted_elbo_components():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(11, 2)).astype(np.float32)
    Y = rng.binomial(1, 0.5, size=(11, 3)).astype(np.float32)
    beta_kl = 0.4
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=3,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=8,
        n_hidden_layers=1,
    )

    history = trainer.fit(
        X,
        Y,
        X_val=X[:5],
        Y_val=Y[:5],
        epochs=2,
        batch_size=4,
        beta_kl=beta_kl,
        kl_warmup_epochs=0,
        verbose=False,
        seed=2026,
    )

    for split in ("train", "val"):
        assert len(history[f"{split}_recon_loss"]) == 2
        assert len(history[f"{split}_kl_loss"]) == 2
        np.testing.assert_allclose(
            history[f"{split}_loss"],
            np.asarray(history[f"{split}_recon_loss"])
            + beta_kl * np.asarray(history[f"{split}_kl_loss"]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            history[f"{split}_recon_per_outcome"],
            np.asarray(history[f"{split}_recon_loss"]) / 3.0,
        )


def test_kl_warmup_and_latent_usage_are_reported():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 2)).astype(np.float32)
    Y = rng.binomial(1, 0.5, size=(20, 3)).astype(np.float32)
    trainer = CVAETrainer(2, 3, latent_dim=2, hidden_dim=8, n_hidden_layers=1)

    history = trainer.fit(
        X,
        Y,
        epochs=3,
        beta_kl=0.9,
        kl_warmup_epochs=3,
        max_grad_norm=5.0,
        verbose=False,
        seed=7,
    )

    np.testing.assert_allclose(history["effective_beta_kl"], [0.3, 0.6, 0.9])
    assert all(values.shape == (2,) for values in history["train_kl_per_latent"])
    assert all(0 <= count <= 2 for count in history["active_latent_units"])


def test_early_stopping_restores_best_checkpoint():
    rng = np.random.default_rng(88)
    X = rng.normal(size=(24, 2)).astype(np.float32)
    Y = rng.binomial(1, 0.5, size=(24, 3)).astype(np.float32)

    torch.manual_seed(99)
    one_epoch = CVAETrainer(2, 3, latent_dim=2, hidden_dim=8, n_hidden_layers=1)
    one_epoch.fit(
        X[4:],
        Y[4:],
        X_val=X[:4],
        Y_val=Y[:4],
        epochs=1,
        batch_size=5,
        verbose=False,
        seed=99,
    )

    torch.manual_seed(99)
    stopped = CVAETrainer(2, 3, latent_dim=2, hidden_dim=8, n_hidden_layers=1)
    history = stopped.fit(
        X[4:],
        Y[4:],
        X_val=X[:4],
        Y_val=Y[:4],
        epochs=5,
        batch_size=5,
        early_stopping_patience=1,
        early_stopping_min_delta=1e9,
        verbose=False,
        seed=99,
    )

    assert history["best_epoch"] == 1
    assert history["epochs_ran"] == 2
    for name, expected in one_epoch.model.state_dict().items():
        torch.testing.assert_close(stopped.model.state_dict()[name], expected)


def test_early_stopping_selection_and_patience_begin_at_requested_epoch():
    rng = np.random.default_rng(188)
    X = rng.normal(size=(24, 2)).astype(np.float32)
    Y = rng.binomial(1, 0.5, size=(24, 3)).astype(np.float32)
    start_epoch = 3
    patience = 2

    trainer = CVAETrainer(2, 3, latent_dim=2, hidden_dim=8, n_hidden_layers=1)
    history = trainer.fit(
        X[4:],
        Y[4:],
        X_val=X[:4],
        Y_val=Y[:4],
        epochs=8,
        batch_size=5,
        early_stopping_patience=patience,
        early_stopping_min_delta=1e9,
        early_stopping_start_epoch=start_epoch,
        verbose=False,
        seed=188,
    )

    # The first eligible epoch always establishes the checkpoint. With an
    # unattainable subsequent improvement, exactly `patience` more epochs run.
    assert history["early_stopping_start_epoch"] == start_epoch
    assert history["best_epoch"] == start_epoch
    assert history["epochs_ran"] == start_epoch + patience


def test_early_stopping_start_cannot_exceed_epochs_with_validation_data():
    rng = np.random.default_rng(288)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    Y = rng.binomial(1, 0.5, size=(12, 3)).astype(np.float32)
    trainer = CVAETrainer(2, 3, latent_dim=2, hidden_dim=8, n_hidden_layers=1)

    with pytest.raises(ValueError, match="cannot exceed num_epochs"):
        trainer.fit(
            X[4:],
            Y[4:],
            X_val=X[:4],
            Y_val=Y[:4],
            epochs=2,
            early_stopping_start_epoch=3,
            verbose=False,
        )
