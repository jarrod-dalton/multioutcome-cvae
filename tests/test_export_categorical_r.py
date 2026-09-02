import shutil
import subprocess

import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer
from multioutcome_cvae.export_r import export_categorical_r


SCHEMA = [
    {"name": "status", "levels": ["off", "on"]},
    {"name": "group", "levels": ["alpha", "beta", "gamma"]},
    {
        "name": "rating",
        "levels": ["very low", "low", "middle", "high", "very high"],
    },
]


def _fitted_trainer():
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=3,
        latent_dim=2,
        outcome_type="categorical",
        outcome_schema=SCHEMA,
        hidden_dim=6,
        n_hidden_layers=1,
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


def test_export_requires_fitted_categorical_model_and_consistent_metadata(tmp_path):
    with pytest.raises(TypeError, match="CVAETrainer"):
        export_categorical_r(object(), tmp_path / "model.R")

    trainer = CVAETrainer(
        2,
        3,
        outcome_type="categorical",
        outcome_schema=SCHEMA,
    )
    with pytest.raises(RuntimeError, match="fitted before export"):
        export_categorical_r(trainer, tmp_path / "model.R")

    bernoulli = CVAETrainer(2, 3, outcome_type="bernoulli")
    bernoulli.x_mean = np.zeros(2, dtype=np.float32)
    bernoulli.x_std = np.ones(2, dtype=np.float32)
    bernoulli.trained = True
    with pytest.raises(ValueError, match="categorical models only"):
        export_categorical_r(bernoulli, tmp_path / "model.R")

    trainer = _fitted_trainer()
    with pytest.raises(ValueError, match="feature_names must contain exactly 2"):
        export_categorical_r(
            trainer,
            tmp_path / "model.R",
            feature_names=["only_one"],
        )
    with pytest.raises(ValueError, match="feature_names must not contain duplicates"):
        export_categorical_r(
            trainer,
            tmp_path / "model.R",
            feature_names=["same", "same"],
        )

    trainer.outcome_slices = [(0, 2), (2, 5), (5, 9)]
    with pytest.raises(ValueError, match="slices are inconsistent"):
        export_categorical_r(trainer, tmp_path / "model.R")

    trainer = _fitted_trainer()
    trainer.model.outcome_schema[0]["levels"] = ["on", "off"]
    with pytest.raises(ValueError, match="model metadata does not match"):
        export_categorical_r(trainer, tmp_path / "model.R")


def test_export_is_format_two_base_r_source(tmp_path):
    output_path = export_categorical_r(
        _fitted_trainer(),
        tmp_path / "nested" / "model.R",
        feature_names=["age", "score"],
    )
    source = output_path.read_text(encoding="utf-8")
    assert output_path == tmp_path / "nested" / "model.R"
    assert "format_version = 2L" in source
    assert "cvae_model_metadata <- function()" in source
    assert "cvae_decoder_probabilities <- function(X, Z)" in source
    assert "cvae_marginal_probabilities <- function(" in source
    assert "cvae_simulate <- function(" in source
    assert "library(" not in source
    assert "require(" not in source
    assert "reticulate" not in source


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_exported_categorical_r_decoder_matches_python_for_fixed_x_and_z(tmp_path):
    torch.manual_seed(321)
    trainer = _fitted_trainer()
    model_path = export_categorical_r(
        trainer,
        tmp_path / "model.R",
        feature_names=["age", "score"],
    )

    X = np.array([[1.0, -0.5], [2.0, 0.25]], dtype=np.float32)
    Z = np.array([[0.1, -1.0], [0.75, 0.5]], dtype=np.float32)
    X_std = (X - trainer.x_mean) / trainer.x_std
    with torch.no_grad():
        logits = trainer.model.decode(
            torch.from_numpy(X_std), torch.from_numpy(Z)
        )["logits"]
        expected = np.concatenate(
            [
                torch.softmax(logits[:, start:stop], dim=1).numpy().ravel()
                for start, stop in trainer.outcome_slices
            ]
        )

    script_path = tmp_path / "parity.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE)
colnames(X) <- c("age", "score")
Z <- matrix(c(0.1, -1.0, 0.75, 0.5), nrow = 2L, byrow = TRUE)
p <- cvae_decoder_probabilities(X, Z)
values <- unlist(lapply(p, function(value) as.vector(t(value))), use.names = FALSE)
cat(paste(format(values, digits = 17L, scientific = TRUE), collapse = ","))
''',
        encoding="utf-8",
    )
    actual = np.fromstring(_run_r(script_path).stdout, sep=",")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_exported_categorical_metadata_order_and_feature_reordering(tmp_path):
    model_path = export_categorical_r(
        _fitted_trainer(),
        tmp_path / "model.R",
        feature_names=["age", "score"],
    )
    script_path = tmp_path / "metadata.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
metadata <- cvae_model_metadata()
stopifnot(
  identical(metadata$format_version, 2L),
  identical(metadata$outcome_type, "categorical"),
  identical(metadata$encoded_y_dim, 10L),
  identical(names(metadata$outcome_schema), NULL),
  identical(vapply(metadata$outcome_schema, `[[`, character(1L), "name"),
            c("status", "group", "rating")),
  identical(metadata$outcome_schema[[2L]]$levels, c("alpha", "beta", "gamma")),
  identical(unname(metadata$outcome_slices),
            matrix(c(0L, 2L, 2L, 5L, 5L, 10L), nrow = 3L, byrow = TRUE)),
  identical(colnames(metadata$outcome_slices), c("start", "stop")),
  identical(rownames(metadata$outcome_slices), c("status", "group", "rating"))
)
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE,
            dimnames = list(NULL, c("age", "score")))
Z <- matrix(c(0.1, -1.0, 0.75, 0.5), nrow = 2L, byrow = TRUE)
ordered <- cvae_decoder_probabilities(X, Z)
reordered <- cvae_decoder_probabilities(X[, c("score", "age"), drop = FALSE], Z)
stopifnot(
  identical(names(ordered), c("status", "group", "rating")),
  identical(lapply(ordered, colnames),
            list(status = c("off", "on"),
                 group = c("alpha", "beta", "gamma"),
                 rating = c("very low", "low", "middle", "high", "very high"))),
  all(vapply(ordered, function(value) all(abs(rowSums(value) - 1) < 1e-12), logical(1L))),
  isTRUE(all.equal(ordered, reordered, tolerance = 0))
)
''',
        encoding="utf-8",
    )
    _run_r(script_path)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_exported_categorical_simulation_is_seeded_valid_and_batched(tmp_path):
    model_path = export_categorical_r(_fitted_trainer(), tmp_path / "model.R")
    script_path = tmp_path / "simulate.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE)

decoder <- cvae_decoder_probabilities
largest_batch <- 0L
cvae_decoder_probabilities <- function(X, Z) {{
  largest_batch <<- max(largest_batch, nrow(X))
  decoder(X, Z)
}}

a <- cvae_simulate(X, n_samples_per_x = 7L, seed = 99L, decoder_batch_size = 3L)
b <- cvae_simulate(X, n_samples_per_x = 7L, seed = 99L, decoder_batch_size = 3L)
stopifnot(
  identical(a, b),
  identical(dim(a), c(2L, 7L, 3L)),
  identical(typeof(a), "integer"),
  identical(dimnames(a)[[3L]], c("status", "group", "rating")),
  all(a[, , 1L] %in% 0:1),
  all(a[, , 2L] %in% 0:2),
  all(a[, , 3L] %in% 0:4),
  largest_batch <= 3L
)

largest_batch <- 0L
marginal <- cvae_marginal_probabilities(
  X[1L, , drop = FALSE], n_mc = 7L, seed = 42L, decoder_batch_size = 3L
)
stopifnot(
  identical(names(marginal), c("status", "group", "rating")),
  identical(vapply(marginal, nrow, integer(1L)), c(status = 1L, group = 1L, rating = 1L)),
  identical(vapply(marginal, ncol, integer(1L)), c(status = 2L, group = 3L, rating = 5L)),
  all(vapply(marginal, function(value) abs(sum(value) - 1) < 1e-12, logical(1L))),
  largest_batch <= 3L
)

single <- cvae_simulate(X[1L, , drop = FALSE], seed = 7L, decoder_batch_size = 1L)
stopifnot(
  identical(dim(single), c(1L, 3L)),
  identical(typeof(single), "integer"),
  identical(colnames(single), c("status", "group", "rating"))
)
''',
        encoding="utf-8",
    )
    _run_r(script_path)
