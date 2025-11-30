# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate binary outcomes**.

---

# Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation / Environment](#installation--environment)
- [1. Native Python Usage](#1-native-python-usage)
- [2. RStudio / Posit Workbench Usage (via reticulate)](#2-rstudio--posit-workbench-usage-via-reticulate)
- [3. Databricks Usage](#3-databricks-usage)
    - [3.1 Python Notebooks](#31-python-notebooks)
    - [3.2 R Notebooks](#32-r-notebooks)
- [4. MLflow Example](#4-mlflow-example)
- [5. Evaluation Notes](#5-evaluation-notes)
- [6. Self-Contained Example: `fit_cvae_with_tuning()` + MLflow (Databricks)](#6-selfcontained-example-fit_cvae_with_tuning--mlflow-databricks)

---

# Overview

This repository provides tools to learn the conditional distribution:

- **X** = covariates  
- **Y** = vector of binary outcomes (e.g., 10 correlated Bernoulli variables)

via a **Conditional Variational Autoencoder (CVAE)**.

The CVAE allows you to:

1. Estimate **p(Y | X)**  
2. Generate **realistic, correlated Y vectors** for any covariate profile  
3. Capture latent dependence across multivariate binary outcomes  

Includes:

- Data simulation (`simulate_cvae_data`)
- Flexible neural architecture (configurable hidden layers)
- Hyperparameter tuning:
  - Random search  
  - TPE (Tree-structured Parzen Estimator) via `hyperopt`
- Log-likelihood evaluation
- Correlation heatmaps
- MLflow training example
- Full R / reticulate integration
- A convenience function `fit_cvae_with_tuning()` for non-MLflow workflows

---

# Repository Structure

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py
│       ├── model.py             # CVAE model, trainer, tuning, fit_cvae_with_tuning()
│       ├── simulate.py          # data simulation + correlation utilities
│       └── examples
│           └── train_with_mlflow.py
├── R
│   └── fit_cvae_example.R
└── README.md
```

---

# Installation / Environment

Can be used from:

- Native Python
- RStudio / Posit Workbench (`reticulate`)
- Databricks (Python or R notebooks)

### Python dependencies

Core:
- `numpy`
- `torch`
- `matplotlib`

Optional:
- `mlflow` (only if tracking/logging)
- `hyperopt` (only for TPE tuning)

---

# 1. Native Python Usage

```python
import sys
sys.path.append("/path/to/repo/python")

from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    fit_cvae_with_tuning,
    compare_real_vs_generated
)

X, Y, params = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234,
)

search_space = {
    "latent_dim":      [4, 8, 16],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40],
}

fit_res = fit_cvae_with_tuning(
    X=X,
    Y=Y,
    search_space=search_space,
    method="random",
    n_trials=10,
    train_frac=0.8,
    seed=1234,
)

trainer = fit_res["trainer"]
Y_gen = trainer.generate(X[:5], n_samples_per_x=10)
```

---

# 2. RStudio / Posit Workbench Usage (via `reticulate`)

```r
library(reticulate)
mb <- import("multibin_cvae")

sim <- mb$simulate_cvae_data(
  n_samples=20000L,
  n_features=5L,
  n_outcomes=10L,
  latent_dim=2L,
  seed=1234L
)

X <- sim[[1]]
Y <- sim[[2]]

search_space <- dict(
  latent_dim=list(4L, 8L),
  enc_hidden_dims=list(list(64L,64L), list(128L,64L)),
  dec_hidden_dims=list(list(64L,64L), list(128L,64L)),
  lr=list(1e-3,5e-4),
  beta_kl=list(0.5,1.0),
  batch_size=list(128L,256L),
  num_epochs=list(20L,40L)
)

fit_res <- mb$fit_cvae_with_tuning(
  X=X, Y=Y,
  search_space=search_space,
  method="random",
  n_trials=5L
)

trainer <- fit_res[["trainer"]]
Y_gen <- trainer$generate(X[1:5,,drop=FALSE], n_samples_per_x=10L)
```

---

# 3. Databricks Usage

## 3.1 Python Notebooks

```python
import sys
sys.path.append("/Workspace/Users/<your_user>/cvae_multibin/python")

from multibin_cvae import simulate_cvae_data, fit_cvae_with_tuning
```

Then train normally.

## 3.2 R Notebooks

Use a `%python` cell first:

```python
import sys
sys.path.append("/Workspace/Users/<your_user>/cvae_multibin/python")
```

Then in R:

```r
mb <- reticulate::import("multibin_cvae")
```

---

# 4. MLflow Example

See:
```
python/multibin_cvae/examples/train_with_mlflow.py
```

It demonstrates:

- Data simulation  
- CVAE training  
- Logging:
  - hyperparameters  
  - train/val losses  
  - correlation heatmap  
  - PyTorch model  
  - standardization parameters  

---

# 5. Evaluation Notes

Two evaluation views:

### 1. Log-likelihood / cross-entropy  
`CVAETrainer.evaluate_loglik(X, Y, ...)` computes:

\[
\sum_{i,j} \left[ y_{ij}\log(p_{ij}) + (1-y_{ij})\log(1-p_{ij}) \right]
\]

with an MC estimate of \(p_{ij}\).

### 2. Correlation structure  
`compare_real_vs_generated()`  
`summarize_binary_matrix()`

Both provide masked-diagonal heatmaps for clarity.

---

# 6. Self-Contained Example: `fit_cvae_with_tuning()` + MLflow (Databricks)

This is a **minimal reproducible** Databricks Python example that:

1. Simulates data  
2. Performs hyperparameter tuning  
3. Refits the best model on all data  
4. Logs the model + metrics + correlation heatmap to MLflow  

Paste the whole block into a Databricks **Python notebook**:

```python
import sys
sys.path.append("/Workspace/Users/<your_user>/cvae_multibin/python")

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np

from multibin_cvae import (
    simulate_cvae_data,
    fit_cvae_with_tuning,
    compare_real_vs_generated
)

# -------------------------------------------------------
# 1. Simulate data
# -------------------------------------------------------
X, Y, params = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234,
)

# -------------------------------------------------------
# 2. Define search space for hyperparameter tuning
# -------------------------------------------------------
search_space = {
    "latent_dim":      [4, 8],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40],
}

# -------------------------------------------------------
# 3. Run tuning + refit best model
# -------------------------------------------------------
fit_res = fit_cvae_with_tuning(
    X=X,
    Y=Y,
    search_space=search_space,
    method="random",  # or "tpe" if hyperopt installed
    n_trials=8,
    train_frac=0.8,
    seed=1234,
    verbose=True,
)

trainer = fit_res["trainer"]
best_config = fit_res["best_config"]

# -------------------------------------------------------
# 4. MLflow logging
# -------------------------------------------------------
experiment_name = "/Users/<your_user>/CVAE_fit_tuning"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_params(best_config)

    # Log standardization attributes
    mlflow.log_dict(
        {"x_mean": trainer.x_mean.tolist(), "x_std": trainer.x_std.tolist()},
        "standardization.json"
    )

    # Log the trained PyTorch model
    mlflow.pytorch.log_model(trainer.model, artifact_path="cvae_model")

    # Compare real vs generated
    Y_gen = trainer.generate(X[:2000], n_samples_per_x=1)
    _, ax = compare_real_vs_generated(Y_real=Y[:2000], Y_gen=Y_gen, make_plot=True)

    # Log heatmap
    fig = ax.get_figure()
    mlflow.log_figure(fig, "correlation_comparison.png")
    plt.close(fig)

    # Log evaluation metrics
    metrics = trainer.evaluate_loglik(X=X[:2000], Y=Y[:2000], n_mc=20)
    mlflow.log_metrics(metrics)

print("Finished MLflow run.")
```

This pipeline gives you:

- Automated hyperparameter search  
- Best-config model fit  
- Correlation heatmap saved as MLflow artifact  
- Log-likelihood metrics tracked  
- Reusable model in the MLflow Model Registry  

---

That’s the complete README.
