"""
Example: Train CVAE with MLflow logging, including:
- model + preprocessing
- log-likelihood-based goodness-of-fit
- correlation heatmaps (via mlflow.log_figure)
"""

import sys
import numpy as np
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# Adjust to your workspace path
sys.path.append("/Workspace/Users/daltonj@ccf.org/cvae_multibin/python")

from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    CVAETrainer,
    compare_real_vs_generated,
)


def main():
    # 1. Set experiment
    mlflow.set_experiment("/Users/daltonj@ccf.org/experiments/multibin_cvae")

    # 2. Simulate data
    X, Y, params_true = simulate_cvae_data(
        n_samples=20000,
        n_features=5,
        n_outcomes=10,
        latent_dim=2,
        seed=1234,
    )

    splits = train_val_test_split(X, Y, train_frac=0.7, val_frac=0.15, seed=1234)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val,   Y_val   = splits["X_val"],   splits["Y_val"]
    X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

    # 3. Hyperparameters for this run
    latent_dim = 8
    enc_hidden_dims = [128, 64]
    dec_hidden_dims = [128, 64]
    num_epochs = 40
    batch_size = 256
    lr = 1e-3
    beta_kl = 1.0

    with mlflow.start_run(run_name="cvae_with_loglik_and_heatmaps") as run:
        # 4. Log parameters
        mlflow.log_params({
            "latent_dim": latent_dim,
            "enc_hidden_dims": enc_hidden_dims,
            "dec_hidden_dims": dec_hidden_dims,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "beta_kl": beta_kl,
            "x_dim": X_train.shape[1],
            "y_dim": Y_train.shape[1],
        })

        # 5. Train CVAE
        trainer = CVAETrainer(
            x_dim=X_train.shape[1],
            y_dim=Y_train.shape[1],
            latent_dim=latent_dim,
            enc_hidden_dims=enc_hidden_dims,
            dec_hidden_dims=dec_hidden_dims,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            beta_kl=beta_kl,
        )

        history = trainer.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            beta_kl=beta_kl,
            verbose=True,
            seed=1234,
        )

        train_loss_final = history["train_loss"][-1]
        val_loss_final = history["val_loss"][-1] if history["val_loss"] else train_loss_final

        mlflow.log_metrics({
            "train_loss_final": train_loss_final,
            "val_loss_final": val_loss_final,
        })

        # 6. Evaluate on test set: log-likelihood-based GOF
        ll_metrics = trainer.evaluate_loglik(X_test, Y_test, n_mc=20)
        mlflow.log_metrics({
            "test_sum_loglik": ll_metrics["sum_loglik"],
            "test_avg_loglik": ll_metrics["avg_loglik"],
            "test_avg_bce": ll_metrics["avg_bce"],
        })

        # 7. Generate Y for test X and compare correlations; log heatmap
        Y_gen = trainer.generate(X_test, n_samples_per_x=10, return_probs=False)
        Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

        res = compare_real_vs_generated(
            Y_real=Y_test,
            Y_gen=Y_gen_flat,
            label_real="Test",
            label_gen="Generated",
            make_plot=True,
            show=False,  # we'll let mlflow handle saving, not interactive display
        )

        fig = res.get("fig", None)
        if fig is not None:
            mlflow.log_figure(fig, "corr_heatmaps_test_vs_generated.png")
            plt.close(fig)

        # 8. Log model + preprocessing
        mlflow.pytorch.log_model(
            trainer.model,
            artifact_path="model",
        )

        np.savez("x_standardizer.npz",
                 x_mean=trainer.x_mean,
                 x_std=trainer.x_std)
        mlflow.log_artifact("x_standardizer.npz", artifact_path="preprocessing")

        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()
