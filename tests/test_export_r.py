import shutil
import subprocess

import numpy as np
import pytest
import torch

from multioutcome_cvae import CVAETrainer, export_bernoulli_r


def _fitted_trainer():
    trainer = CVAETrainer(
        x_dim=2,
        y_dim=3,
        latent_dim=2,
        outcome_type="bernoulli",
        hidden_dim=6,
        n_hidden_layers=1,
    )
    trainer.x_mean = np.array([0.5, -0.25], dtype=np.float32)
    trainer.x_std = np.array([2.0, 0.5], dtype=np.float32)
    trainer.trained = True
    return trainer


def test_export_requires_fitted_bernoulli_model(tmp_path):
    trainer = CVAETrainer(2, 3, outcome_type="bernoulli")
    with pytest.raises(RuntimeError, match="fitted before export"):
        export_bernoulli_r(trainer, tmp_path / "model.R")

    gaussian = CVAETrainer(2, 3, outcome_type="gaussian")
    gaussian.trained = True
    gaussian.x_mean = np.zeros(2, dtype=np.float32)
    gaussian.x_std = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="Bernoulli models only"):
        export_bernoulli_r(gaussian, tmp_path / "model.R")


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_exported_r_decoder_matches_python_for_fixed_x_and_z(tmp_path):
    torch.manual_seed(321)
    trainer = _fitted_trainer()
    model_path = export_bernoulli_r(
        trainer,
        tmp_path / "model.R",
        feature_names=["age", "score"],
        outcome_names=["a", "b", "c"],
    )

    X = np.array([[1.0, -0.5], [2.0, 0.25]], dtype=np.float32)
    Z = np.array([[0.1, -1.0], [0.75, 0.5]], dtype=np.float32)
    X_std = (X - trainer.x_mean) / trainer.x_std
    with torch.no_grad():
        expected = torch.sigmoid(
            trainer.model.decode(torch.from_numpy(X_std), torch.from_numpy(Z))["logits"]
        ).numpy()

    script_path = tmp_path / "parity.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE)
colnames(X) <- c("age", "score")
Z <- matrix(c(0.1, -1.0, 0.75, 0.5), nrow = 2L, byrow = TRUE)
p <- cvae_decoder_probabilities(X, Z)
cat(paste(format(as.vector(t(p)), digits = 17L, scientific = TRUE), collapse = ","))
''',
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["Rscript", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = np.fromstring(completed.stdout, sep=",").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_exported_r_simulation_is_seeded_and_has_expected_shape(tmp_path):
    model_path = export_bernoulli_r(_fitted_trainer(), tmp_path / "model.R")
    script_path = tmp_path / "simulate.R"
    script_path.write_text(
        f'''source({str(model_path)!r})
X <- matrix(c(1.0, -0.5, 2.0, 0.25), nrow = 2L, byrow = TRUE)
a <- cvae_simulate(X, n_samples_per_x = 4L, seed = 99L, inference_batch_size = 1L)
b <- cvae_simulate(X, n_samples_per_x = 4L, seed = 99L, inference_batch_size = 1L)
stopifnot(identical(a, b), identical(dim(a), c(2L, 4L, 3L)), all(a %in% c(0L, 1L)))
''',
        encoding="utf-8",
    )
    subprocess.run(["Rscript", str(script_path)], check=True, capture_output=True, text=True)