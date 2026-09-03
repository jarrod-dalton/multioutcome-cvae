import shutil
import subprocess

import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer, export_categorical_r
from multioutcome_cvae.model import (
    PRODUCTION_OUTCOME_TYPES,
    VALIDATION_PENDING_OUTCOME_TYPES,
    VALID_OUTCOME_TYPES,
)


def test_categorical_is_api_supported_but_predictive_validation_pending():
    assert "categorical" in VALID_OUTCOME_TYPES
    assert "categorical" in VALIDATION_PENDING_OUTCOME_TYPES
    assert "categorical" not in PRODUCTION_OUTCOME_TYPES


def _ready_trainer(seed, schema):
    torch.manual_seed(seed)
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=len(schema),
        latent_dim=2,
        outcome_type="categorical",
        outcome_schema=schema,
        hidden_dim=6,
        n_hidden_layers=1,
        device="cpu",
    )
    trainer.x_mean = np.array([0.5, -0.25], dtype=np.float32)
    trainer.x_std = np.array([2.0, 0.5], dtype=np.float32)
    trainer.trained = True
    return trainer


def _run_r(script_path):
    return subprocess.run(
        ["Rscript", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_categorical_r_v2_enforces_embedded_feature_names(tmp_path):
    schema = [
        {"name": "flag", "levels": ["off", "on"]},
        {"name": "group", "levels": ["a", "b", "c"]},
    ]
    model_path = export_categorical_r(
        _ready_trainer(101, schema),
        tmp_path / "categorical.R",
        feature_names=["age", "score"],
    )
    script_path = tmp_path / "feature_contract.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
Z <- matrix(c(0.25, -0.5, 0.75, 0.1), nrow = 2L, byrow = TRUE)
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE,
            dimnames = list(NULL, c("age", "score")))

ordered <- cvae_decoder_probabilities(X, Z)
reordered <- cvae_decoder_probabilities(X[, c("score", "age"), drop = FALSE], Z)
expect_error <- function(thunk) inherits(try(thunk(), silent = TRUE), "try-error")

unnamed <- unname(X)
missing <- X
colnames(missing) <- c("age", "not_score")
duplicated <- X
colnames(duplicated) <- c("age", "age")

stopifnot(
  isTRUE(all.equal(ordered, reordered, tolerance = 0)),
  expect_error(function() cvae_decoder_probabilities(unnamed, Z)),
  expect_error(function() cvae_decoder_probabilities(missing, Z)),
  expect_error(function() cvae_decoder_probabilities(duplicated, Z))
)
''',
        encoding="utf-8",
    )

    _run_r(script_path)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_two_categorical_r_models_coexist_in_isolated_environments(tmp_path):
    schema_a = [
        {"name": "status_a", "levels": ["off", "on"]},
        {"name": "group_a", "levels": ["a", "b", "c"]},
    ]
    schema_b = [
        {"name": "status_b", "levels": ["no", "yes"]},
        {"name": "group_b", "levels": ["low", "mid", "high"]},
    ]
    model_a = export_categorical_r(
        _ready_trainer(201, schema_a),
        tmp_path / "model_a.R",
        feature_names=["x1", "x2"],
    )
    model_b = export_categorical_r(
        _ready_trainer(202, schema_b),
        tmp_path / "model_b.R",
        feature_names=["x1", "x2"],
    )
    script_path = tmp_path / "isolated_models.R"
    script_path.write_text(
        f'''model_a <- new.env(parent = baseenv())
model_b <- new.env(parent = baseenv())
sys.source({str(model_a)!r}, envir = model_a)

X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE,
            dimnames = list(NULL, c("x1", "x2")))
Z <- matrix(c(0.25, -0.5, 0.75, 0.1), nrow = 2L, byrow = TRUE)
a_before <- model_a$cvae_decoder_probabilities(X, Z)

sys.source({str(model_b)!r}, envir = model_b)
a_after <- model_a$cvae_decoder_probabilities(X, Z)
b <- model_b$cvae_decoder_probabilities(X, Z)
sim_a <- model_a$cvae_simulate(X, n_samples_per_x = 2L, seed = 31L)
sim_b <- model_b$cvae_simulate(X, n_samples_per_x = 2L, seed = 32L)

stopifnot(
  identical(names(a_before), c("status_a", "group_a")),
  identical(names(b), c("status_b", "group_b")),
  isTRUE(all.equal(a_before, a_after, tolerance = 0)),
  !isTRUE(all.equal(unname(a_after), unname(b), tolerance = 0)),
  identical(dim(sim_a), c(2L, 2L, 2L)),
  identical(dim(sim_b), c(2L, 2L, 2L)),
  identical(dimnames(sim_a)[[3L]], c("status_a", "group_a")),
  identical(dimnames(sim_b)[[3L]], c("status_b", "group_b"))
)
''',
        encoding="utf-8",
    )

    _run_r(script_path)
