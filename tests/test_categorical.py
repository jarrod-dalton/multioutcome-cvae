import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from multioutcome_cvae import CVAETrainer, MultivariateOutcomeCVAE
from multioutcome_cvae.model import (
    fit_cvae_with_tuning,
    tune_cvae_random_search,
    tune_cvae_tpe,
)


SCHEMA = [
    {"name": "binary", "levels": ["no", "yes"]},
    {"name": "three", "levels": ["a", "b", "c"]},
    {"name": "five", "levels": ["0", "1", "2", "3", "4+"]},
]


def _ready_trainer(schema=SCHEMA):
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=len(schema),
        latent_dim=2,
        outcome_type="categorical",
        hidden_dim=8,
        n_hidden_layers=1,
        device="cpu",
        outcome_schema=schema,
    )
    trainer._fit_standardizer(np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32))
    trainer.trained = True
    return trainer


def test_schema_is_normalized_copied_and_drives_dimensions():
    supplied = [
        {"name": item["name"], "levels": list(item["levels"])} for item in SCHEMA
    ]
    trainer = _ready_trainer(supplied)

    assert trainer.y_dim == 3
    assert trainer.encoded_y_dim == 10
    assert trainer.outcome_slices == [(0, 2), (2, 5), (5, 10)]
    assert trainer.model.enc_layers[0].in_features == 2 + 10
    assert trainer.model.dec_logits_head[-1].out_features == 10
    assert trainer.model.dec_logits_skip.out_features == 10

    supplied[0]["name"] = "changed"
    supplied[1]["levels"].append("changed")
    assert trainer.outcome_schema == SCHEMA
    assert trainer.model.outcome_schema == SCHEMA


@pytest.mark.parametrize(
    "schema, match",
    [
        (None, "required"),
        (SCHEMA[:2], "exactly one entry"),
        (
            [SCHEMA[0], {"name": "binary", "levels": ["x", "y"]}, SCHEMA[2]],
            "Duplicate outcome name",
        ),
        (
            [SCHEMA[0], {"name": "three", "levels": ["x", "x"]}, SCHEMA[2]],
            "duplicate level",
        ),
        (
            [SCHEMA[0], {"name": "three", "levels": ["only"]}, SCHEMA[2]],
            "at least two levels",
        ),
        (
            [SCHEMA[0], {"name": "", "levels": ["x", "y"]}, SCHEMA[2]],
            "non-empty string",
        ),
        (
            [SCHEMA[0], {"name": "three", "levels": ["x", ""]}, SCHEMA[2]],
            "non-empty string",
        ),
    ],
)
def test_invalid_categorical_schemas_are_rejected(schema, match):
    with pytest.raises(ValueError, match=match):
        CVAETrainer(
            x_dim=2,
            y_dim=3,
            outcome_type="categorical",
            outcome_schema=schema,
        )


def test_schema_is_rejected_for_existing_families():
    with pytest.raises(ValueError, match="only valid"):
        CVAETrainer(
            x_dim=2,
            y_dim=3,
            outcome_type="bernoulli",
            outcome_schema=SCHEMA,
        )


def test_categorical_encoder_uses_one_hot_groups_and_zeroes_masked_group():
    model = MultivariateOutcomeCVAE(
        x_dim=2,
        y_dim=3,
        latent_dim=2,
        outcome_type="categorical",
        enc_hidden_dims=[8],
        dec_hidden_dims=[8],
        outcome_schema=SCHEMA,
    )
    y = torch.tensor([[1.0, 2.0, 4.0], [0.0, 1.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    encoded = model._encode_categorical_y(y, mask=mask)

    assert encoded.shape == (2, 10)
    assert torch.equal(encoded[0, :2], torch.tensor([0.0, 1.0]))
    assert torch.equal(encoded[0, 2:5], torch.zeros(3))
    assert torch.equal(encoded[0, 5:10], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

    y_changed_under_mask = y.clone()
    y_changed_under_mask[0, 1] = 0
    x = torch.zeros((2, 2))
    mu_1, logvar_1 = model.encode(x, y, mask=mask)
    mu_2, logvar_2 = model.encode(x, y_changed_under_mask, mask=mask)
    assert torch.equal(mu_1, mu_2)
    assert torch.equal(logvar_1, logvar_2)


@pytest.mark.parametrize(
    "bad_value, column, match",
    [
        (0.5, 0, "integers"),
        (1.0 + 1e-8, 0, "integers"),
        (-1.0, 1, "between 0 and 2"),
        (3.0, 1, "between 0 and 2"),
        (np.nan, 2, "finite"),
    ],
)
def test_fit_rejects_invalid_codes_even_when_masked(bad_value, column, match):
    trainer = _ready_trainer()
    X = np.zeros((4, 2), dtype=np.float32)
    Y = np.zeros((4, 3), dtype=np.float64)
    Y[0, column] = bad_value
    mask = np.ones_like(Y)
    mask[0, column] = 0

    with pytest.raises(ValueError, match=match):
        trainer.fit(X, Y, Y_mask_train=mask, epochs=1, verbose=False)


@pytest.mark.parametrize(
    "mask, match",
    [
        (np.ones((4, 2)), "shape"),
        (np.array([[1, 1, 1], [1, 0.5, 1], [1, 1, 1], [1, 1, 1]]), "0 or 1"),
        (np.array([[1, 1, 1], [1, np.nan, 1], [1, 1, 1], [1, 1, 1]]), "finite"),
    ],
)
def test_fit_strictly_validates_categorical_masks(mask, match):
    trainer = _ready_trainer()
    X = np.zeros((4, 2), dtype=np.float32)
    Y = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        trainer.fit(X, Y, Y_mask_train=mask, epochs=1, verbose=False)


def test_categorical_loss_is_summed_per_semantic_outcome_and_masked():
    trainer = _ready_trainer()
    y = torch.tensor([[0.0, 2.0, 4.0], [1.0, 0.0, 3.0]])
    logits = torch.tensor(
        [
            [0.1, -0.1, 0.2, 0.5, -0.4, 0.1, 0.2, 0.3, 0.4, 0.5],
            [-0.2, 0.8, 0.4, -0.3, 0.1, 0.7, -0.2, 0.0, 0.6, 0.1],
        ]
    )
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    observed_loss = trainer._recon_loss(y, {"logits": logits}, mask=mask)
    manual = logits.new_zeros(())
    for index, (start, stop) in enumerate(trainer.outcome_slices):
        manual += (
            F.cross_entropy(
                logits[:, start:stop], y[:, index].long(), reduction="none"
            )
            * mask[:, index]
        ).sum()
    assert torch.allclose(observed_loss, manual)


def test_two_class_softmax_and_cross_entropy_reduce_to_bernoulli():
    logits = torch.tensor(
        [[-2.0, 1.0], [0.5, -0.25], [3.0, 3.0]], dtype=torch.float64
    )
    targets = torch.tensor([1, 0, 1])
    bernoulli_logits = logits[:, 1] - logits[:, 0]

    assert torch.allclose(
        torch.softmax(logits, dim=1)[:, 1], torch.sigmoid(bernoulli_logits)
    )
    assert torch.allclose(
        F.cross_entropy(logits, targets, reduction="sum"),
        F.binary_cross_entropy_with_logits(
            bernoulli_logits, targets.to(torch.float64), reduction="sum"
        ),
    )


def test_mixed_cardinality_training_prediction_and_generation_contracts():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(24, 2)).astype(np.float32)
    Y = np.column_stack(
        [
            rng.integers(0, 2, size=24),
            rng.integers(0, 3, size=24),
            rng.integers(0, 5, size=24),
        ]
    )
    mask = np.ones_like(Y, dtype=np.float32)
    mask[::3, 1] = 0
    trainer = CVAETrainer(
        2,
        3,
        latent_dim=2,
        outcome_type="categorical",
        hidden_dim=8,
        n_hidden_layers=1,
        batch_size=8,
        outcome_schema=SCHEMA,
        device="cpu",
    )
    history = trainer.fit(
        X, Y, Y_mask_train=mask, epochs=1, verbose=False, seed=77
    )
    assert np.isfinite(history["train_loss"]).all()

    params = trainer.predict_params(X[:5], n_mc=2, inference_batch_size=2)
    assert list(params) == ["probabilities"]
    assert list(params["probabilities"]) == ["binary", "three", "five"]
    for name, cardinality in zip(("binary", "three", "five"), (2, 3, 5)):
        probabilities = params["probabilities"][name]
        assert probabilities.shape == (5, cardinality)
        assert probabilities.dtype == np.float32
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    samples = trainer.generate(X[:5], n_samples_per_x=4)
    assert samples.shape == (5, 4, 3)
    assert samples.dtype == np.int32
    for index, cardinality in enumerate((2, 3, 5)):
        assert np.all((samples[..., index] >= 0) & (samples[..., index] < cardinality))

    returned_probabilities = trainer.generate(
        X[:5], n_samples_per_x=2, return_probs=True
    )
    assert list(returned_probabilities) == ["binary", "three", "five"]
    with pytest.raises(ValueError, match="not defined"):
        trainer.predict_mean(X[:2])
    with pytest.raises(ValueError, match="only defined"):
        trainer.predict_proba(X[:2])


def test_all_binary_schema_and_decoder_batch_cap():
    schema = [
        {"name": "a", "levels": ["0", "1"]},
        {"name": "b", "levels": ["0", "1"]},
    ]
    rng = np.random.default_rng(902)
    X = rng.normal(size=(16, 2)).astype(np.float32)
    Y = rng.integers(0, 2, size=(16, 2), dtype=np.int32)
    torch.manual_seed(902)
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=2,
        latent_dim=2,
        outcome_type="categorical",
        hidden_dim=8,
        n_hidden_layers=1,
        device="cpu",
        outcome_schema=schema,
    )
    history = trainer.fit(X, Y, epochs=1, batch_size=4, verbose=False, seed=902)
    assert np.isfinite(history["train_loss"]).all()
    probabilities = trainer.predict_params(X[:3], n_mc=2)["probabilities"]
    assert [matrix.shape for matrix in probabilities.values()] == [(3, 2), (3, 2)]

    decoder_batch_sizes = []
    original_decode = trainer.model.decode

    def recording_decode(x, z):
        decoder_batch_sizes.append(x.shape[0])
        return original_decode(x, z)

    trainer.model.decode = recording_decode
    samples = trainer.generate(
        X[:7],
        n_samples_per_x=5,
        inference_batch_size=4,
        decoder_batch_size=3,
    )
    assert samples.shape == (7, 5, 2)
    assert samples.dtype == np.int32
    assert np.isin(samples, (0, 1)).all()
    assert max(decoder_batch_sizes) <= 3

    with pytest.raises(ValueError, match="positive integer"):
        trainer.generate(X[:1], decoder_batch_size=0)


def test_tuning_interfaces_accept_schema_and_random_search_threads_it():
    assert "outcome_schema" in inspect.signature(tune_cvae_random_search).parameters
    assert "outcome_schema" in inspect.signature(tune_cvae_tpe).parameters
    assert "outcome_schema" in inspect.signature(fit_cvae_with_tuning).parameters

    rng = np.random.default_rng(4)
    X = rng.normal(size=(16, 2)).astype(np.float32)
    Y = np.column_stack(
        [rng.integers(0, 2, size=16), rng.integers(0, 3, size=16)]
    )
    results = tune_cvae_random_search(
        X[:12],
        Y[:12],
        X[12:],
        Y[12:],
        x_dim=2,
        y_dim=2,
        search_space={
            "latent_dim": [2],
            "hidden_dim": [8],
            "n_hidden_layers": [1],
            "num_epochs": [1],
            "batch_size": [4],
        },
        n_trials=1,
        outcome_type="categorical",
        outcome_schema=SCHEMA[:2],
        device="cpu",
        verbose=False,
    )
    assert np.isfinite(results["best_val_loss"])
