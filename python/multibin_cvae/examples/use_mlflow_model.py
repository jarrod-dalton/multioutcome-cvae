"""
examples/use_mlflow_model.py

Example of how to:

  - Load a trained CVAE model from an MLflow run
  - Load the associated standardization parameters for X
  - Use the model to generate Y | X for new covariates

Assumptions:
  - train_with_mlflow.py logged the model with artifact_path="model"
  - It also logged x_mean.npy and x_std.npy as artifacts
  - outcome_type is "bernoulli" in this example

Usage:

  export MLFLOW_RUN_ID=<your_run_id>
  python -m multibin_cvae.examples.use_mlflow_model
"""

import os
from typing import Optional

import numpy as np
import mlflow
import mlflow.pytorch
import torch


class LoadedCVAEModule:
    """
    Thin wrapper around the raw CVAE PyTorch module that:

      - stores x_mean, x_std used for standardization
      - exposes generate() and predict_proba() for Bernoulli

    This mirrors the parts of CVAETrainer you need at inference time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.x_std[self.x_std < 1e-8] = 1.0  # guard against zeros

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def generate(
        self,
        X_new: np.ndarray,
        n_samples_per_x: int = 1,
        return_probs: bool = False,
    ) -> np.ndarray:
        """
        Generate from the model's decoder:

          - If return_probs=True (Bernoulli case), returns P(Y_ij = 1 | X_i)
          - Otherwise, returns Bernoulli samples (0/1)

        This assumes the underlying model has:
          - attributes: x_dim, y_dim, latent_dim, outcome_type
          - a .decode(x, z) method that returns {"logits": ...} for Bernoulli
        """
        X_new = np.asarray(X_new, dtype=np.float32)
        X_std = self._standardize(X_new)
        n, x_dim = X_std.shape

        x_tensor = torch.from_numpy(X_std).to(self.device)
        total = n * n_samples_per_x
        x_rep = x_tensor.repeat_interleave(n_samples_per_x, dim=0)

        # Latent z samples
        latent_dim = self.model.latent_dim
        z = torch.randn((total, latent_dim), device=self.device)

        with torch.no_grad():
            out = self.model.decode(x_rep, z)
            # Here we assume Bernoulli, so decoder returns logits
            logits = out["logits"]
            probs = torch.sigmoid(logits)

            if return_probs:
                probs_np = probs.cpu().numpy()
                if n_samples_per_x == 1:
                    return probs_np.reshape(n, self.model.y_dim)
                else:
                    return probs_np.reshape(n, n_samples_per_x, self.model.y_dim)

            bern = torch.distributions.Bernoulli(probs=probs)
            y_samples = bern.sample().cpu().numpy().astype(np.int32)

            if n_samples_per_x == 1:
                return y_samples.reshape(n, self.model.y_dim)
            else:
                return y_samples.reshape(n, n_samples_per_x, self.model.y_dim)

    def predict_proba(
        self,
        X_new: np.ndarray,
        n_mc: int = 20,
    ) -> np.ndarray:
        """
        Approximate P(Y_ij = 1 | X_i) by averaging over multiple
        draws of z (Monte Carlo).
        """
        X_new = np.asarray(X_new, dtype=np.float32)
        X_std = self._standardize(X_new)
        n, x_dim = X_std.shape

        x_tensor = torch.from_numpy(X_std).to(self.device)

        acc = torch.zeros((n, self.model.y_dim), device=self.device)
        with torch.no_grad():
            for _ in range(n_mc):
                z = torch.randn((n, self.model.latent_dim), device=self.device)
                out = self.model.decode(x_tensor, z)
                logits = out["logits"]
                probs = torch.sigmoid(logits)
                acc += probs

        probs_mean = (acc / float(n_mc)).cpu().numpy()
        return probs_mean


def main():
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise ValueError(
            "Please set MLFLOW_RUN_ID env var to the run ID from train_with_mlflow.py"
        )

    print(f"Loading model from MLflow run: {run_id}")

    # -------------------------------
    # 1. Load model and artifacts
    # -------------------------------
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id)
    artifacts_path = "."

    # Download x_mean and x_std artifacts
    x_mean_path = client.download_artifacts(run_id, "x_mean.npy", dst_path=artifacts_path)
    x_std_path = client.download_artifacts(run_id, "x_std.npy", dst_path=artifacts_path)

    x_mean = np.load(x_mean_path)
    x_std = np.load(x_std_path)

    # Wrap the raw model with standardization
    loaded = LoadedCVAEModule(model=model, x_mean=x_mean, x_std=x_std)

    print("Loaded model and standardization parameters.")
    print("  x_mean shape:", x_mean.shape)
    print("  x_std shape:", x_std.shape)

    # -------------------------------
    # 2. Create some new covariates X_new
    # -------------------------------
    rng = np.random.default_rng(999)
    x_dim = model.x_dim
    X_new = rng.normal(size=(5, x_dim)).astype(np.float32)

    # -------------------------------
    # 3. Use the model for prediction / simulation
    # -------------------------------
    probs = loaded.predict_proba(X_new, n_mc=30)
    print("\nPredicted probabilities for first new row:")
    print(probs[0])

    Y_sim = loaded.generate(X_new, n_samples_per_x=5, return_probs=False)
    print("\nGenerated Y_sim shape:", Y_sim.shape)


if __name__ == "__main__":
    main()
