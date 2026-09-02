"""Characterize the legacy Bernoulli/Gaussian boundary for additive changes."""

import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer, MultivariateOutcomeCVAE, export_bernoulli_r


@pytest.mark.parametrize("outcome_type", ["bernoulli", "gaussian"])
def test_legacy_family_architecture_shapes_are_unchanged(outcome_type):
    model = MultivariateOutcomeCVAE(
        x_dim=3,
        y_dim=2,
        latent_dim=4,
        outcome_type=outcome_type,
        enc_hidden_dims=[7, 5],
        dec_hidden_dims=[6],
    )

    assert model.encoded_y_dim == 2
    assert model.outcome_schema is None
    assert model.outcome_slices == []
    assert [(layer.in_features, layer.out_features) for layer in model.enc_layers] == [
        (5, 7),
        (7, 5),
    ]
    assert [(layer.in_features, layer.out_features) for layer in model.dec_layers] == [
        (7, 6)
    ]
    assert (model.enc_mu.in_features, model.enc_mu.out_features) == (5, 4)
    assert (model.enc_logvar.in_features, model.enc_logvar.out_features) == (5, 4)

    if outcome_type == "bernoulli":
        assert model.dec_logits_head[-1].out_features == 2
        assert (model.dec_logits_skip.in_features, model.dec_logits_skip.out_features) == (
            3,
            2,
        )
    else:
        assert model.dec_mu_head[-1].out_features == 2
        assert model.dec_logvar_head[-1].out_features == 2
        assert (model.dec_mu_skip.in_features, model.dec_mu_skip.out_features) == (3, 2)


@pytest.mark.parametrize(
    ("outcome_type", "parameter_keys", "sample_dtype"),
    [
        ("bernoulli", {"probs"}, np.dtype(np.int32)),
        ("gaussian", {"mu", "sigma"}, np.dtype(np.float32)),
    ],
)
def test_legacy_prediction_and_generation_contracts(
    outcome_type, parameter_keys, sample_dtype
):
    rng = np.random.default_rng(880)
    X = rng.normal(size=(18, 3)).astype(np.float32)
    if outcome_type == "bernoulli":
        Y = rng.integers(0, 2, size=(18, 2)).astype(np.float32)
    else:
        Y = rng.normal(size=(18, 2)).astype(np.float32)

    torch.manual_seed(880)
    trainer = CVAETrainer(
        x_dim=3,
        y_dim=2,
        latent_dim=2,
        outcome_type=outcome_type,
        hidden_dim=8,
        n_hidden_layers=1,
        batch_size=6,
        device="cpu",
    )
    trainer.fit(X, Y, epochs=1, verbose=False, seed=880)

    parameters = trainer.predict_params(X[:4], n_mc=2, inference_batch_size=3)
    assert set(parameters) == parameter_keys
    for value in parameters.values():
        assert value.shape == (4, 2)
        assert value.dtype == np.float32

    samples = trainer.generate(X[:4], n_samples_per_x=3, inference_batch_size=2)
    assert samples.shape == (4, 3, 2)
    assert samples.dtype == sample_dtype


def test_bernoulli_r_export_remains_format_version_one(tmp_path):
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=2,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=6,
        n_hidden_layers=1,
        device="cpu",
    )
    trainer.x_mean = np.zeros(2, dtype=np.float32)
    trainer.x_std = np.ones(2, dtype=np.float32)
    trainer.trained = True

    source = export_bernoulli_r(
        trainer,
        tmp_path / "model.R",
        feature_names=["x1", "x2"],
        outcome_names=["y1", "y2"],
    ).read_text(encoding="utf-8")

    assert "format_version = 1L" in source
    assert 'outcome_type = "bernoulli"' in source
    assert "cvae_decoder_probabilities <- function(X, Z)" in source
    assert "cvae_marginal_probabilities <- function(" in source
    assert "cvae_simulate <- function(" in source
    assert "outcome_schema" not in source
    assert "cvae_model_metadata" not in source
