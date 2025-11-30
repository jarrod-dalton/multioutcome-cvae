# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate binary outcomes**.

Goal: learn a flexible model for:

- **X** = covariates / predictors  
- **Y** = vector of binary outcomes (e.g., 10 correlated Bernoulli variables)

and use it to:

1. Estimate **p(Y | X)** (conditional probabilities)  
2. Generate realistic, correlated binary outcomes for new covariates.

---

# Table of Contents
- [Repository structure](#repository-structure)
- [Installation / environment](#installation--environment)
- [1. Native Python usage](#1-native-python-usage)
- [2. RStudio / Posit Workbench usage via reticulate](#2-rstudio--posit-workbench-usage-via-reticulate)
- [3. Databricks usage](#3-databricks-usage)
  - [3a. Databricks Python notebook](#3a-databricks-python-notebook)
  - [3b. Databricks R notebook](#3b-databricks-r-notebook)
- [4. MLflow example](#4-mlflow-example)
- [5. Evaluation notes](#5-evaluation-notes)

---

## Repository structure

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py          # exposes main functions/classes
│       ├── model.py             # CVAE model, trainer, tuning, fit_cvae_with_tuning()
│       ├── simulate.py          # data simulation + correlation utilities
│       └── examples
│           └── train_with_mlflow.py  # MLflow example (Databricks-friendly)
├── R
│   └── fit_cvae_example.R       # R/reticulate usage example
└── README.md
```

The core components are `model.py` and `simulate.py`.

---

## Installation / environment

You can use this code in:

- **Native Python** (local / server)
- **RStudio / Posit Workbench** via `reticulate`
- **Databricks** (Python and/or R notebooks)

The package is “source-only.” Typical options:

- Add the `python/` folder to `PYTHONPATH`
- Or in Python:

```python
import sys
sys.path.append("/path/to/repo/python")
```

### Dependencies

Core:

- `numpy`
- `torch`
- `matplotlib`

Optional:

- `mlflow` (only if using MLflow logging)
- `hyperopt` (only if using TPE tuning)

Install example:

```bash
pip install numpy torch matplotlib mlflow hyperopt
```

---

# 1. Native Python usage

Below shows **both** direct training and the convenience method `fit_cvae_with_tuning()`.

```python
import sys
sys.path.append("/path/to/repo/python")

from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    CVAETrainer,
    fit_cvae_with_tuning,
    compare_real_vs_generated,
)

# --- Simulate data ---
X, Y, params_true = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234,
)

# --- Split data ---
splits = train_val_test_split(X, Y, train_frac=0.7, val_frac=0.15, seed=1234)
X_train, Y_train = splits["X_train"], splits["Y_train"]
X_val,   Y_val   = splits["X_val"],   splits["Y_val"]
X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

# --- Search space for tuning ---
search_space = {
    "latent_dim":      [4, 8],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40],
}

# --- Fit CVAE with tuning ---
fit_res = fit_cvae_with_tuning(
    X=X,
    Y=Y,
    search_space=search_space,
    method="random",   # or "tpe"
    n_trials=10,
    train_frac=0.8,
    seed=1234,
)

trainer = fit_res["trainer"]

# --- Generate new correlated Y for test X ---
X_new = X_test[:5]
Y_sim = trainer.generate(X_new, n_samples_per_x=10)

# --- Evaluate correlation structure ---
Y_sim_flat = Y_sim.reshape(-1, Y_sim.shape[-1])
compare_real_vs_generated(Y_test, Y_sim_flat, make_plot=True)
```

---

# 2. RStudio / Posit Workbench usage via reticulate

This example uses `fit_cvae_with_tuning()` directly from R.

```r
library(reticulate)

# Optional: force exact Python
# use_python("/path/to/python", required = TRUE)

mb <- import("multibin_cvae")

# --- Simulate data ---
sim <- mb$simulate_cvae_data(
  n_samples  = 20000L,
  n_features = 5L,
  n_outcomes = 10L,
  latent_dim = 2L,
  seed       = 1234L
)

X <- sim[[1]]
Y <- sim[[2]]

# --- Search space ---
search_space <- dict(
  latent_dim      = list(4L, 8L),
  enc_hidden_dims = list(list(64L, 64L), list(128L, 64L)),
  dec_hidden_dims = list(list(64L, 64L), list(128L, 64L)),
  lr              = list(1e-3, 5e-4),
  beta_kl         = list(0.5, 1.0),
  batch_size      = list(128L, 256L),
  num_epochs      = list(20L, 40L)
)

# --- Fit with tuning ---
fit_res <- mb$fit_cvae_with_tuning(
  X           = X,
  Y           = Y,
  search_space = search_space,
  method      = "random",
  n_trials    = 5L,
  train_frac  = 0.8,
  seed        = 1234L
)

trainer <- fit_res[["trainer"]]

# --- Generate ---
Y_sim <- trainer$generate(
  X[1:5, , drop = FALSE],
  n_samples_per_x = 10L
)

# --- Evaluate log-likelihood on held-out ---
ll_metrics <- trainer$evaluate_loglik(
  X = X[1:2000, , drop = FALSE],
  Y = Y[1:2000, , drop = FALSE],
  n_mc = 20L
)
print(ll_metrics)
```

Notes:
- Use `py_config()` to validate which Python `reticulate` is using.
- Make sure the repo’s `python/` directory is discoverable by Python.

---

# 3. Databricks usage

## 3a. Databricks Python notebook

```python
import sys
sys.path.append("/Workspace/Users/your.name@org.org/cvae_multibin/python")

from multibin_cvae import (
    simulate_cvae_data,
    fit_cvae_with_tuning,
)

# Simulate data
X, Y, params = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234,
)

# Search space
search_space = {
    "latent_dim": [4, 8],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr": [1e-3, 5e-4],
    "beta_kl": [0.5, 1.0],
    "batch_size": [128, 256],
    "num_epochs": [20, 40],
}

# Fit with tuning
fit_res = fit_cvae_with_tuning(
    X, Y,
    search_space=search_space,
    method="random",
    n_trials=8,
    train_frac=0.8,
    seed=777
)

trainer = fit_res["trainer"]

# Generate
Y_gen = trainer.generate(X[:10], n_samples_per_x=5)
Y_gen
```

---

## 3b. Databricks R notebook

First, ensure Python sees the package:

**Python cell:**

```python
import sys
sys.path.append("/Workspace/Users/your.name@org.org/cvae_multibin/python")
```

**R cell:**

```r
library(reticulate)
py_config()

mb <- import("multibin_cvae")

sim <- mb$simulate_cvae_data(20000L, 5L, 10L, 2L, 1234L)
X <- sim[[1]]; Y <- sim[[2]]

search_space <- dict(
  latent_dim = list(4L, 8L),
  enc_hidden_dims = list(list(64L, 64L), list(128L, 64L)),
  dec_hidden_dims = list(list(64L, 64L), list(128L, 64L)),
  lr = list(1e-3, 5e-4),
  beta_kl = list(0.5, 1.0),
  batch_size = list(128L, 256L),
  num_epochs = list(20L, 40L)
)

fit_res <- mb$fit_cvae_with_tuning(
  X, Y,
  search_space = search_space,
  method = "random",
  n_trials = 5L
)

trainer <- fit_res[["trainer"]]
trainer$generate(X[1:5, , drop=FALSE], n_samples_per_x=10L)
```

---

# 4. MLflow example

A complete MLflow-friendly script is provided:

```
python/multibin_cvae/examples/train_with_mlflow.py
```

It demonstrates:

- Simulating or loading real data  
- Training CVAE  
- Logging:
  - hyperparameters  
  - losses  
  - correlation heatmap (`mlflow.log_figure`)  
  - trained model & standardization parameters  

Adapt it simply by replacing the data-loading block.

---

# 5. Evaluation notes

Two evaluation styles are included:

## 1. Log-likelihood / cross-entropy
`CVAETrainer.evaluate_loglik(X, Y, ...)` computes

\[
\sum_{i,j} y_{ij} \log p_{ij} + (1-y_{ij}) \log(1-p_{ij})
\]

where \(p_{ij}\) is approximated via Monte Carlo over the latent space.

## 2. Correlation fidelity
- `summarize_binary_matrix(Y, make_plot=True)`  
- `compare_real_vs_generated(Y_real, Y_gen, make_plot=True)`

These inspect marginal probabilities and correlation structure to confirm that the CVAE captures multivariate dependencies, not just marginal distributions.

---

End of README.
