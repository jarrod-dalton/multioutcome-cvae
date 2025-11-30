# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate binary outcomes**.

Goal: estimate and simulate from complex conditional distributions

- **X** = covariates / predictors  
- **Y** = vector of binary outcomes (e.g., 10 correlated Bernoulli variables)

so you can:

1. Estimate **p(Y | X)**  
2. Generate **realistic, correlated Y vectors** for new X  
3. Evaluate fit using **log-likelihood** and **correlation structure**

Includes:

- flexible CVAE model + trainer  
- random search and TPE hyperparameter tuning  
- log-likelihood evaluation  
- correlation/marginal diagnostic heatmaps  
- Databricks + MLflow training example  
- R integration via `reticulate`  
- convenience function `fit_cvae_with_tuning()`  

---

# Table of Contents
- [1. Repository Structure](#1-repository-structure)  
- [2. Installation / Environment](#2-installation--environment)  
- [3. Basic Usage Examples (Python, R, Databricks)](#3-basic-usage-examples-python-r-databricks)  
- [4. Full Databricks + MLflow Example (with tuning)](#4-full-databricks--mlflow-example-with-tuning)  
- [5. Evaluation Notes](#5-evaluation-notes)  

---

# 1. Repository Structure

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py
│       ├── model.py             # CVAETrainer, tuning, fit_cvae_with_tuning()
│       ├── simulate.py          # simulate data + correlation utilities
│       └── examples
│           └── train_with_mlflow.py   # (not required anymore; replaced by Section 4)
├── R
│   └── fit_cvae_example.R       # R reticulate usage example
└── README.md
```

Critical modules:  
- `model.py` – CVAE model + trainer + tuning utilities  
- `simulate.py` – data simulation + summary + heatmaps  

---

# 2. Installation / Environment

This code is *source-only* (not on PyPI). You can use it from:

- Local / server Python  
- RStudio / Posit Workbench via **reticulate**  
- Databricks (Python + R notebooks)

### Requirements

Core:
```
numpy
torch
matplotlib
```

Optional:
```
mlflow     # For experiment tracking
hyperopt   # For TPE tuning
```

### Making the package importable
Any of these works:

#### Option A (quick)
```python
import sys
sys.path.append("/path/to/repo/python")
import multibin_cvae
```

#### Option B (editable install)
Add `pyproject.toml` later and run:
```
pip install -e python/
```

---

# 3. Basic Usage Examples (Python, R, Databricks)

## 3.1 Python (local/server)

```python
import sys
sys.path.append("/path/to/repo/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

# simulate example data
X, Y, params = simulate_cvae_data(
    n_samples=5000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=123
)

trainer = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=8,
    enc_hidden_dims=[64,64],
    dec_hidden_dims=[64,64],
)

trainer.fit(X, Y, verbose=True)

Y_new = trainer.generate(X[:10], n_samples_per_x=5)
```

---

## 3.2 RStudio / Posit (reticulate)

```r
library(reticulate)

# Configure Python env if needed:
# use_python("/opt/miniconda/envs/myenv/bin/python", required=TRUE)

mb <- import("multibin_cvae")

sim <- mb$simulate_cvae_data(
  n_samples=5000L,
  n_features=5L,
  n_outcomes=10L,
  latent_dim=2L,
  seed=1234L
)

X <- sim[[1]]
Y <- sim[[2]]

trainer <- mb$CVAETrainer(
  x_dim=ncol(X),
  y_dim=ncol(Y),
  latent_dim=8L
)

trainer$fit(X, Y)
Y_new <- trainer$generate(X[1:5,], n_samples_per_x=10L)
```

---

## 3.3 Databricks (Python notebook)

```python
import sys
sys.path.append("/Workspace/Users/<user>/cvae_multibin/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

X, Y, _ = simulate_cvae_data(20000, 5, 10, latent_dim=2, seed=123)

trainer = CVAETrainer(x_dim=5, y_dim=10, latent_dim=8)
trainer.fit(X, Y)

trainer.generate(X[:5], n_samples_per_x=5)
```

---

# 4. Full Databricks + MLflow Example (with tuning)

This replaces the old “examples/train_with_mlflow.py”.

## 4.1 Notebook setup

```python
import sys
sys.path.append("/Workspace/Users/<user>/cvae_multibin/python")

import mlflow
mlflow.autolog(disable=True)

from multibin_cvae import (
    simulate_cvae_data,
    fit_cvae_with_tuning,
    compare_real_vs_generated
)
```

## 4.2 Simulate (or load real) data

```python
X, Y, params = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234
)
```

## 4.3 Define search space (discrete)

```python
search_space = {
    "latent_dim":      [4, 8, 16],
    "enc_hidden_dims": [[64,64], [128,64]],
    "dec_hidden_dims": [[64,64], [128,64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40]
}
```

## 4.4 Run tuning + final training inside an MLflow run

```python
import mlflow

with mlflow.start_run():

    mlflow.log_param("tuning_method", "random")
    mlflow.log_param("n_trials", 8)

    results = fit_cvae_with_tuning(
        X=X,
        Y=Y,
        search_space=search_space,
        method="random",
        n_trials=8,
        train_frac=0.8,
        seed=1234,
        verbose=True
    )

    trainer = results["trainer"]
    best_cfg = results["best_config"]

    mlflow.log_params(best_cfg)

    # Evaluate
    import numpy as np
    test_idx = np.random.choice(X.shape[0], 3000, replace=False)
    metrics = trainer.evaluate_loglik(X[test_idx], Y[test_idx])

    mlflow.log_metrics(metrics)

    # Correlation heatmap
    from multibin_cvae import compare_real_vs_generated
    import matplotlib.pyplot as plt

    Y_gen = trainer.generate(X[test_idx], n_samples_per_x=1)
    Y_gen = Y_gen.reshape(-1, Y_gen.shape[-1])

    fig = compare_real_vs_generated(Y[test_idx], Y_gen, make_plot=True)
    mlflow.log_figure(fig, "correlation_heatmap.png")

    # Save model
    mlflow.pytorch.log_model(trainer.model, "model")
```

You now have a reusable CVAE training pipeline with tracking, parameters, metrics, and artifact storage.

---

# 5. Evaluation Notes

## 5.1 Log-likelihood / cross-entropy

Given:

- \( Y \in \{0,1\}^{n \times d} \)
- CVAE posterior-averaged probabilities  
  \( \hat{p}_{ij} = \mathbb{E}_z[p(Y_{ij}=1 \mid X_i, z)] \)

The log-likelihood-style measure is:

\[
\text{LL} = \sum_{i=1}^n \sum_{j=1}^d
\left[
Y_{ij} \log(\hat{p}_{ij}) +
(1 - Y_{ij}) \log(1 - \hat{p}_{ij})
\right].
\]

Average per-element loglik:

\[
\text{avg\_LL} = \frac{1}{nd} \text{LL}.
\]

Negative average loglik = cross-entropy:

\[
\text{avg\_BCE} = -\text{avg\_LL}.
\]

Call:

```python
trainer.evaluate_loglik(X_test, Y_test)
```

## 5.2 Correlation diagnostics

Heatmaps compare dependence structure:

- `summarize_binary_matrix(Y, make_plot=True)`  
- `compare_real_vs_generated(Y_real, Y_gen, make_plot=True)`  

These visualize:

- Marginal prevalence  
- Pairwise correlation structure (diagonal masked)

Useful for checking joint behavior not captured by log-likelihood alone.

---
