"""
examples/train_with_mlflow.py

End-to-end example:

  1. Simulate multivariate Bernoulli data (X, Y)
  2. Tune CVAE hyperparameters on a train/validation split
  3. Refit the best model on the full dataset
  4. Evaluate log-likelihood on a test subset
  5. Compare correlation structure (real vs generated)
  6. Log everything to MLflow

Intended usage:

    python -m multioutcome_cvae.examples.train_with_mlflow

In Databricks, you can also copy/paste the contents of main() into a
Python notebook cell, or `import` this file and call run_experiment().
"""

import os
import tempfile
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

from multioutcome_cvae import (
    simulate_cvae_data,
    fit_cvae_with_tuning,
    compare_real_vs_generated,
)


def run_experiment(
    n_samples: int = 20000,
    n_features: int = 5,
    n_outcomes: int = 10,
    latent_dim: int = 2,
    outcome_type: str = "bernoulli",
    n_trials: int = 8,
    train_frac: float = 0.8,
    seed: int = 1234,
    tuning_method: str = "random",  # "random" or "tpe" (if hyperopt installed)
    experiment_name: str = "multioutcome_cvae_mlflow",
    run_name: str = "cvae_tuned_bernoulli",
) -> Dict[str, Any]:
    """
    Run a single MLflow experiment: simulate -> tune -> fit -> evaluate -> log.

    Parameters
    ----------
    n_samples : int, default=20000
        Number of observations to simulate.

    n_features : int, default=5
        Number of covariates (columns of X).

    n_outcomes : int, default=10
        Number of outcome dimensions (columns of Y).

    latent_dim : int, default=2
        Latent dimension used in the simulated data.

    outcome_type : str, default="bernoulli"
        Outcome family for the simulation; this example is designed around
        "bernoulli", but simulate_cvae_data supports "gaussian" and "poisson"
        as well.

    n_trials : int, default=8
        Number of hyperparameter configurations to evaluate.

    train_frac : float, default=0.8
        Fraction of data used for training (the rest is held out internally
        as validation during tuning).

    seed : int, default=1234
        Base random seed for reproducibility.

    tuning_method : {"random", "tpe"}, default="random"
        Hyperparameter search strategy. "tpe" requires hyperopt.

    experiment_name : str, default="multioutcome_cvae_mlflow"
        Name of the MLflow experiment.

    run_name : str, default="cvae_tuned_bernoulli"
        Name of the MLflow run.

    Returns
    -------
    result : dict
        Dictionary with keys:
          - "trainer"        : fitted CVAETrainer
          - "best_config"    : best hyperparameter configuration
          - "tuning_results" : full tuning results
          - "metrics"        : evaluation metrics dict
          - "run_id"         : MLflow run ID
    """
    mlflow.autolog(disable=True)
    mlflow.set_experiment(experiment_name)

    # Search space for tuning
    search_space = {
        "latent_dim":      [4, 8, 16],
        "enc_hidden_dims": [[64, 64], [128, 64]],
        "dec_hidden_dims": [[64, 64], [128, 64]],
        "lr":              [1e-3, 5e-4],
        "beta_kl":         [0.5, 1.0],
        "batch_size":      [128, 256],
        "num_epochs":      [20, 40],
        "hidden_dim":      [64],       # used only if enc/dec_hidden_dims is None
        "n_hidden_layers": [2],        # idem
    }

    with mlflow.start_run(run_name=run_name) as run:

        # -------------------------------
        # 1. Simulate data
        # -------------------------------
        X, Y, params_true = simulate_cvae_data(
            n_samples=n_samples,
            n_features=n_features,
            n_outcomes=n_outcomes,
            latent_dim=latent_dim,
            outcome_type=outcome_type,
            seed=seed,
        )

        mlflow.log_param("sim_outcome_type", outcome_type)
        mlflow.log_param("sim_latent_dim", latent_dim)
        mlflow.log_param("sim_seed", seed)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_outcomes", n_outcomes)

        # -------------------------------
        # 2–3. Tuning + final fit
        # -------------------------------
        mlflow.log_param("tuning_method", tuning_method)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("train_frac", train_frac)

        res = fit_cvae_with_tuning(
            X=X,
            Y=Y,
            search_space=search_space,
            method=tuning_method,
            n_trials=n_trials,
            train_frac=train_frac,
            outcome_type=outcome_type,
            seed=seed,
            verbose=True,
        )

        trainer = res["trainer"]
        best_cfg = res["best_config"]
        tuning_results = res["tuning_results"]

        # Log best hyperparameters
        if best_cfg is not None:
            mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()})

        # -------------------------------
        # 4. Evaluate on a held-out test subset
        # -------------------------------
        rng = np.random.default_rng(seed + 1)
        test_idx = rng.choice(X.shape[0], size=min(3000, X.shape[0]), replace=False)
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        metrics = trainer.evaluate_loglik(X_test, Y_test, n_mc=30)
        mlflow.log_metrics(metrics)

        # -------------------------------
        # 5. Correlation comparison
        # -------------------------------
        Y_gen = trainer.generate(
            X_test,
            n_samples_per_x=1,
            return_probs=False
        )
        Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

        comp = compare_real_vs_generated(
            Y_real=Y_test,
            Y_gen=Y_gen_flat,
            label_real="test",
            label_gen="generated",
            make_plot=True,
        )

        fig = comp.get("fig", None)
        if fig is not None:
            mlflow.log_figure(fig, "corr_test_vs_generated.png")
            plt.close(fig)

        # -------------------------------
        # 5b. Log standardization parameters for X
        # -------------------------------
        # These are needed at inference time to reproduce X -> X_std used in training.
        # They will be consumed by examples/use_mlflow_model.py.
        if trainer.x_mean is not None and trainer.x_std is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                x_mean_path = os.path.join(tmpdir, "x_mean.npy")
                x_std_path = os.path.join(tmpdir, "x_std.npy")

                np.save(x_mean_path, trainer.x_mean)
                np.save(x_std_path, trainer.x_std)

                # Log into root of artifact store (".")
                mlflow.log_artifact(x_mean_path, artifact_path=".")
                mlflow.log_artifact(x_std_path, artifact_path=".")

        # -------------------------------
        # 6. Log model artifact
        # -------------------------------
        mlflow.pytorch.log_model(trainer.model, artifact_path="model")

        run_id = run.info.run_id
        print("MLflow run completed. Run ID:", run_id)

    return {
        "trainer": trainer,
        "best_config": best_cfg,
        "tuning_results": tuning_results,
        "metrics": metrics,
        "run_id": run_id,
    }


def main():
    """
    Simple entry point for running the example as a script.

    You can customize the defaults here or swap in environment variables,
    e.g. via os.getenv("MLFLOW_EXPERIMENT_NAME", "multioutcome_cvae_mlflow").
    """
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "multioutcome_cvae_mlflow")
    run_name = os.getenv("MLFLOW_RUN_NAME", "cvae_tuned_bernoulli")

    result = run_experiment(
        n_samples=20000,
        n_features=5,
        n_outcomes=10,
        latent_dim=2,
        outcome_type="bernoulli",
        n_trials=8,
        train_frac=0.8,
        seed=1234,
        tuning_method="random",
        experiment_name=experiment_name,
        run_name=run_name,
    )

    print("Best config:", result["best_config"])
    print("Metrics:", result["metrics"])
    print("Run ID:", result["run_id"])


if __name__ == "__main__":
    main()
