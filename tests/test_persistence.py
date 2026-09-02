import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer
from multioutcome_cvae.persistence import _safe_torch_load, load_cvae, save_cvae


def _torch_load_for_test(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fit_small_trainer(outcome_type, outcome_schema=None):
    rng = np.random.default_rng(711)
    X = rng.normal(size=(16, 3)).astype(np.float32)
    if outcome_type == "bernoulli":
        Y = rng.integers(0, 2, size=(16, 2)).astype(np.float32)
    elif outcome_type == "gaussian":
        Y = rng.normal(size=(16, 2)).astype(np.float32)
    elif outcome_type == "poisson":
        Y = rng.poisson(1.5, size=(16, 2)).astype(np.float32)
    elif outcome_type == "categorical":
        Y = np.column_stack(
            [rng.integers(0, len(entry["levels"]), size=16) for entry in outcome_schema]
        ).astype(np.float32)
    else:  # pragma: no cover - test helper guard
        raise AssertionError("Unsupported test outcome family")

    kwargs = {}
    if outcome_schema is not None:
        kwargs["outcome_schema"] = outcome_schema
    trainer = CVAETrainer(
        x_dim=3,
        y_dim=2,
        latent_dim=2,
        outcome_type=outcome_type,
        enc_hidden_dims=[7, 5],
        dec_hidden_dims=[6],
        num_epochs=4,
        batch_size=9,
        lr=2e-3,
        beta_kl=0.25,
        device="cpu",
        **kwargs,
    )
    trainer.fit(X, Y, epochs=1, verbose=False, seed=901)
    return trainer, X


def _fixed_decoder_outputs(trainer, X):
    X_std = torch.from_numpy(trainer._standardize(X[:4])).to(trainer.device)
    z = torch.linspace(
        -0.75, 0.75, steps=4 * trainer.latent_dim, device=trainer.device
    ).reshape(4, trainer.latent_dim)
    trainer.model.eval()
    with torch.no_grad():
        return {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.decode(X_std, z).items()
        }


@pytest.mark.parametrize("outcome_type", ["bernoulli", "gaussian"])
def test_checkpoint_round_trip_preserves_fixed_decoder(outcome_type, tmp_path):
    trainer, X = _fit_small_trainer(outcome_type)
    expected = _fixed_decoder_outputs(trainer, X)

    checkpoint = save_cvae(trainer, tmp_path / "nested" / f"{outcome_type}.pt")
    restored = load_cvae(checkpoint, device="cpu")

    assert checkpoint == tmp_path / "nested" / f"{outcome_type}.pt"
    assert restored.trained is True
    assert restored.device == torch.device("cpu")
    assert restored.outcome_type == outcome_type
    assert restored.x_dim == trainer.x_dim
    assert restored.y_dim == trainer.y_dim
    assert restored.latent_dim == trainer.latent_dim
    assert restored.enc_hidden_dims == [7, 5]
    assert restored.dec_hidden_dims == [6]
    assert restored.num_epochs == 4
    assert restored.batch_size == 9
    assert restored.lr == pytest.approx(2e-3)
    assert restored.beta_kl == pytest.approx(0.25)
    np.testing.assert_array_equal(restored.x_mean, trainer.x_mean)
    np.testing.assert_array_equal(restored.x_std, trainer.x_std)

    actual = _fixed_decoder_outputs(restored, X)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_checkpoint_payload_is_minimal_and_versioned(tmp_path):
    trainer, _ = _fit_small_trainer("bernoulli")
    checkpoint = save_cvae(trainer, tmp_path / "model.pt")
    payload = _torch_load_for_test(checkpoint)
    metadata = payload["metadata"]

    assert set(payload) == {
        "format_version",
        "metadata",
        "state_dict",
        "x_mean",
        "x_std",
    }
    assert payload["format_version"] == 1
    assert isinstance(payload["metadata"], dict)
    assert metadata["artifact_type"] == "multioutcome-cvae"
    assert metadata["outcome"] == {
        "type": "bernoulli",
        "outcome_schema": None,
        "encoded_y_dim": 2,
        "outcome_slices": None,
    }
    assert all(torch.is_tensor(value) for value in payload["state_dict"].values())
    assert torch.is_tensor(payload["x_mean"])
    assert torch.is_tensor(payload["x_std"])


def test_categorical_checkpoint_preserves_schema_layout_and_decoder(tmp_path):
    schema = [
        {"name": "flag", "levels": ["no", "yes"]},
        {"name": "status", "levels": ["low", "middle", "high"]},
    ]
    trainer, X = _fit_small_trainer("categorical", schema)
    expected = _fixed_decoder_outputs(trainer, X)

    checkpoint = save_cvae(trainer, tmp_path / "categorical.pt")
    payload = _torch_load_for_test(checkpoint)
    metadata = payload["metadata"]
    assert metadata["outcome"] == {
        "type": "categorical",
        "outcome_schema": schema,
        "encoded_y_dim": 5,
        "outcome_slices": [[0, 2], [2, 5]],
    }

    restored = load_cvae(checkpoint, device="cpu")
    assert restored.outcome_schema == schema
    assert restored.encoded_y_dim == 5
    assert restored.outcome_slices == [(0, 2), (2, 5)]
    actual = _fixed_decoder_outputs(restored, X)
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0.0, atol=0.0)


def test_categorical_checkpoint_rejects_divergent_model_schema(tmp_path):
    schema = [
        {"name": "flag", "levels": ["no", "yes"]},
        {"name": "status", "levels": ["low", "middle", "high"]},
    ]
    trainer, _ = _fit_small_trainer("categorical", schema)
    trainer.model.outcome_schema[1]["levels"] = ["low", "high", "middle"]

    with pytest.raises(RuntimeError, match="model.outcome_schema"):
        save_cvae(trainer, tmp_path / "categorical.pt")


@pytest.mark.parametrize(
    ("constructor_kwargs", "expected_encoder", "expected_decoder"),
    [
        ({"enc_hidden_dims": [], "dec_hidden_dims": []}, [64, 64], [64, 64]),
        ({"n_hidden_layers": 0}, [64, 64], [64, 64]),
        ({"enc_hidden_dims": [5], "dec_hidden_dims": []}, [5], [5]),
    ],
)
def test_checkpoint_records_instantiated_legacy_architecture(
    tmp_path, constructor_kwargs, expected_encoder, expected_decoder
):
    trainer = CVAETrainer(
        2,
        2,
        latent_dim=2,
        outcome_type="bernoulli",
        device="cpu",
        **constructor_kwargs,
    )
    trainer.x_mean = np.zeros(2, dtype=np.float32)
    trainer.x_std = np.ones(2, dtype=np.float32)
    trainer.trained = True

    restored = load_cvae(save_cvae(trainer, tmp_path / "legacy.pt"), device="cpu")
    assert restored.enc_hidden_dims == expected_encoder
    assert restored.dec_hidden_dims == expected_decoder
    assert restored.model.enc_hidden_dims == expected_encoder
    assert restored.model.dec_hidden_dims == expected_decoder


def test_checkpoint_derives_architecture_from_layers_not_mutable_input_lists(tmp_path):
    encoder_dims = [5]
    decoder_dims = [6]
    trainer = CVAETrainer(
        2,
        2,
        latent_dim=2,
        outcome_type="bernoulli",
        enc_hidden_dims=encoder_dims,
        dec_hidden_dims=decoder_dims,
        device="cpu",
    )
    encoder_dims[0] = 50
    decoder_dims[0] = 60
    trainer.x_mean = np.zeros(2, dtype=np.float32)
    trainer.x_std = np.ones(2, dtype=np.float32)
    trainer.trained = True

    restored = load_cvae(save_cvae(trainer, tmp_path / "model.pt"), device="cpu")
    assert restored.enc_hidden_dims == [5]
    assert restored.dec_hidden_dims == [6]


def test_restricted_loader_does_not_retry_after_type_error(monkeypatch, tmp_path):
    calls = []

    def failing_restricted_load(path, map_location=None, weights_only=None):
        calls.append(weights_only)
        raise TypeError("malformed checkpoint")

    monkeypatch.setattr(torch, "load", failing_restricted_load)
    with pytest.raises(TypeError, match="malformed checkpoint"):
        _safe_torch_load(tmp_path / "model.pt", torch.device("cpu"))
    assert calls == [True]


def test_loader_uses_legacy_path_only_when_signature_requires_it(monkeypatch, tmp_path):
    sentinel = object()
    calls = []

    def legacy_load(path, map_location=None):
        calls.append((path, map_location))
        return sentinel

    monkeypatch.setattr(torch, "load", legacy_load)
    path = tmp_path / "model.pt"
    assert _safe_torch_load(path, torch.device("cpu")) is sentinel
    assert calls == [(path, torch.device("cpu"))]


def test_checkpoint_supports_deprecated_poisson_family(tmp_path):
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer, X = _fit_small_trainer("poisson")
    expected = _fixed_decoder_outputs(trainer, X)

    checkpoint = save_cvae(trainer, tmp_path / "poisson.pt")
    with pytest.warns(FutureWarning, match="deprecated"):
        restored = load_cvae(checkpoint, device="cpu")
    actual = _fixed_decoder_outputs(restored, X)
    torch.testing.assert_close(
        actual["log_rate"], expected["log_rate"], rtol=0.0, atol=0.0
    )


def test_save_requires_a_fitted_trainer(tmp_path):
    trainer = CVAETrainer(2, 2, outcome_type="bernoulli", device="cpu")
    with pytest.raises(RuntimeError, match="must be fitted"):
        save_cvae(trainer, tmp_path / "unfitted.pt")
    assert not (tmp_path / "unfitted.pt").exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(format_version=99), "Unsupported.*format_version"),
        (lambda payload: payload.update(metadata="not-a-mapping"), "must be a mapping"),
        (lambda payload: payload.update(metadata={}), "invalid artifact_type"),
    ],
)
def test_load_rejects_malformed_format_or_metadata(tmp_path, mutation, match):
    trainer, _ = _fit_small_trainer("bernoulli")
    checkpoint = save_cvae(trainer, tmp_path / "valid.pt")
    payload = _torch_load_for_test(checkpoint)
    mutation(payload)
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)

    with pytest.raises(ValueError, match=match):
        load_cvae(malformed, device="cpu")


def test_load_recomputes_and_rejects_categorical_layout_metadata(tmp_path):
    schema = [
        {"name": "binary", "levels": ["a", "b"]},
        {"name": "ternary", "levels": ["a", "b", "c"]},
    ]
    trainer, _ = _fit_small_trainer("categorical", schema)
    checkpoint = save_cvae(trainer, tmp_path / "valid-categorical.pt")
    payload = _torch_load_for_test(checkpoint)
    metadata = payload["metadata"]
    metadata["outcome"]["outcome_slices"] = [[0, 2], [2, 4]]
    malformed = tmp_path / "bad-layout.pt"
    torch.save(payload, malformed)

    with pytest.raises(ValueError, match="outcome_slices.*outcome_schema"):
        load_cvae(malformed, device="cpu")
