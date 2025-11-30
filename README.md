# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate outcomes**.

The main use case (and all quick-start examples) are for:

- **X** = covariates / predictors  
- **Y** = vector of **binary outcomes** (Bernoulli)  

so you can:

1. Estimate **p(Y | X)** (via conditional probabilities)  
2. Generate **realistic, correlated Y vectors** for new X  
3. Evaluate fit using a log-likelihood-style score and correlation diagnostics  

The code also supports **Gaussian** and **Poisson** outcome families; those are described in a separate section later.

---

# Table of Contents

- [1. Repository Structure](#1-repository-structure)  
- [2. Installation / Environment](#2-installation--environment)  
- [3. Basic Usage Examples (Bernoulli only)](#3-basic-usage-examples-bernoulli-only)  
  - [3.1 Python](#31-python)  
  - [3.2 R / Posit (reticulate)](#32-r--posit-reticulate)  
  - [3.3 Databricks (Python)](#33-databricks-python)  
- [4. Evaluation Notes (Bernoulli)](#4-evaluation-notes-bernoulli)  
- [5. Other Outcome Families (Gaussian, Poisson)](#5-other-outcome-families-gaussian-poisson)  
- [6. Databricks + MLflow Example (tuning + logging)](#6-databricks--mlflow-example-tuning--logging)  

---

# 1. Repository Structure

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py
│       ├── model.py             # CVAETrainer, tuning, fit_cvae_with_tuning()
│       ├── simulate.py          # data simulation + correlation utilities
│       └── examples             # optional extra scripts (not required)
├── R
│   └── fit_cvae_example.R       # R/reticulate usage example
└── README.md
```

Key modules:

- `model.py` – CVAE model + trainer + tuning utilities  
- `simulate.py` – simulated Bernoulli data + marginal/correlation summaries  

---

# 2. Installation / Environment

This code is **source-only** (not on PyPI). You can use it from:

- Python (local/server)
- RStudio / Posit Workbench via **reticulate**
- Databricks (Python and R notebooks)

## 2.1 Python dependencies

Core:

- `numpy`
- `torch`
- `matplotlib`

Optional:

- `mlflow`   – for experiment tracking & model logging
- `hyperopt` – for TPE tuning

Example installation:

```bash
pip install numpy torch matplotlib mlflow hyperopt
```

## 2.2 Making `multibin_cvae` importable

### Option A (quick)

```python
import sys
sys.path.append("/path/to/your/repo/python")

import multibin_cvae
```

### Option B (editable install, later)

After adding packaging metadata (e.g., `pyproject.toml` or `setup.cfg`):

```bash
pip install -e python/
```

---

# 3. Basic Usage Examples (Bernoulli only)

This section shows **minimal, Bernoulli-only** examples for:

- Python
- R via `reticulate`
- Databricks

Support for other outcome families (Gaussian, Poisson) is discussed in [Section 5](#5-other-outcome-families-gaussian-poisson).

---

## 3.1 Python

```python
import sys
sys.path.append("/path/to/your/repo/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

# Simulate multivariate Bernoulli outcomes Y
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
    outcome_type="bernoulli",      # default; explicit here for clarity
    hidden_dim=64,
    n_hidden_layers=2
)

history = trainer.fit(X, Y, verbose=True)

# Posterior-averaged probabilities p_ij = P(Y_ij = 1 | X_i)
p_hat = trainer.predict_proba(X[:10], n_mc=30)

# Simulate Bernoulli Y given these covariates
Y_sim = trainer.generate(X[:10], n_samples_per_x=5, return_probs=False)
```

---

## 3.2 R / Posit (reticulate)

```r
library(reticulate)

# Optional: pin environment
# use_python("/opt/miniconda/envs/cvae/bin/python", required = TRUE)

mb <- import("multibin_cvae")

# Simulate Bernoulli data in Python, returned to R as arrays
sim <- mb$simulate_cvae_data(
  n_samples  = 5000L,
  n_features = 5L,
  n_outcomes = 10L,
  latent_dim = 2L,
  seed       = 1234L
)

X <- sim[[1]]
Y <- sim[[2]]

trainer <- mb$CVAETrainer(
  x_dim        = ncol(X),
  y_dim        = ncol(Y),
  latent_dim   = 8L,
  outcome_type = "bernoulli"
)

history <- trainer$fit(X, Y)

# Conditional probabilities P(Y_ij = 1 | X_i)
p_hat <- trainer$predict_proba(X[1:10, , drop = FALSE], n_mc = 30L)

# Simulate new Bernoulli outcomes for these covariates
Y_sim <- trainer$generate(
  X[1:5, , drop = FALSE],
  n_samples_per_x = 10L,
  return_probs    = FALSE
)
```

---

## 3.3 Databricks (Python)

```python
import sys
sys.path.append("/Workspace/Users/<user>/cvae_multibin/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

# Simulate Bernoulli data
X, Y, _ = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234
)

trainer = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=8,
    outcome_type="bernoulli"
)

trainer.fit(X, Y, verbose=True)

# Probabilities and generated outcomes
p_hat = trainer.predict_proba(X[:10], n_mc=30)
Y_sim = trainer.generate(X[:5], n_samples_per_x=5)
```

---

# 4. Evaluation Notes (Bernoulli)

The current evaluation tools are designed for **multivariate Bernoulli** Y.

There are two main diagnostics:

1. A **log-likelihood / cross-entropy style** measure based on the posterior-averaged probabilities.
2. **Correlation and marginal structure** comparisons between real and generated Y.

## 4.1 Log-likelihood-style metric (Bernoulli)

For Bernoulli Y, let:

- `Y` be an n × d matrix of 0/1 outcomes.
- `p_hat[i,j]` be the posterior-averaged probability  
  `P(Y[i,j] = 1 | X[i])`, approximated via Monte Carlo over the latent `z`.

We define:

```text
LL = sum_{i=1..n} sum_{j=1..d} [
        Y[i,j]     * log(p_hat[i,j]) +
        (1 - Y[i,j]) * log(1 - p_hat[i,j])
    ]

avg_LL  = LL / (n * d)
avg_BCE = -avg_LL   # "binary cross-entropy" style
```

You can compute these with:

```python
metrics = trainer.evaluate_loglik(X_test, Y_test, n_mc=30)

# metrics = {
#   "sum_loglik": ...,
#   "avg_loglik": ...,
#   "avg_bce"   : ...
# }
```

Currently, `evaluate_loglik()` is implemented for `outcome_type="bernoulli"` only.  
For other families, you can use predictive means / variances and correlation diagnostics as described in the next section.

---

## 4.2 Correlation and marginal structure

Regardless of outcome family, it is important to check:

- Marginal distributions of each component of Y  
- Pairwise correlation structure (off-diagonal elements)

For Bernoulli Y, you can use utilities in `simulate.py` (exposed through `__init__.py`) such as:

```python
from multibin_cvae import summarize_binary_matrix, compare_real_vs_generated

# Inspect real data
summ_real = summarize_binary_matrix(Y_real, name="real", make_plot=True)

# Generate 1 draw per X; flatten into one big matrix
Y_gen = trainer.generate(X_real, n_samples_per_x=1, return_probs=False)
Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

summ = compare_real_vs_generated(
    Y_real=Y_real,
    Y_gen=Y_gen_flat,
    label_real="real",
    label_gen="generated",
    make_plot=True
)
```

These create:

- Marginal prevalence plots  
- Side-by-side correlation heatmaps (diagonal masked)  

and help assess whether the CVAE is capturing **joint dependence**, not just marginal means.

---

# 5. Other Outcome Families (Gaussian, Poisson)

Beyond Bernoulli Y, the core `CVAETrainer` supports:

- `outcome_type="gaussian"` – for continuous Y  
- `outcome_type="poisson"`  – for count Y  

The decoder and loss functions adapt accordingly, and the trainer provides a general **distribution-parameter API** via `predict_params()`.

## 5.1 Decoder parameterization

Given `(X, Z)`, the decoder models:

- Bernoulli:
  - outputs logits `logit_ij`
  - `P(Y_ij = 1 | X_i, Z) = sigmoid(logit_ij)`
- Gaussian:
  - outputs mean `mu_ij` and log-variance `logvar_ij`
  - `Y_ij | X_i, Z ∼ Normal(mu_ij, exp(logvar_ij))`
- Poisson:
  - outputs log-rate `log_rate_ij`
  - `Y_ij | X_i, Z ∼ Poisson(lambda_ij)` where `lambda_ij = exp(log_rate_ij)`

The training loss uses the corresponding negative log-likelihoods (up to constants):

- Bernoulli: binary cross-entropy  
- Gaussian: 0.5 * [logvar + (y - mu)^2 / exp(logvar)]  
- Poisson:  rate - y * log_rate  

---

## 5.2 `predict_params()`: a GLM-like interface

To mirror the idea of `predict.glm(type="link"/"response")`, the trainer provides:

```python
params = trainer.predict_params(X_test, n_mc=20)
```

This returns a dictionary with predictive distribution parameters for **Y | X**, marginalizing over the latent Z by Monte Carlo:

- For Bernoulli:
  ```python
  params = {
      "probs": p_hat  # shape (n, d)
  }
  ```
- For Gaussian:
  ```python
  params = {
      "mu":    mu_pred,    # predictive mean, shape (n, d)
      "sigma": sigma_pred  # predictive SD,   shape (n, d)
  }
  ```
  where `mu_pred` and `sigma_pred` use the mixture-of-Gaussians moment formulas:
  - mu_pred ≈ E[mu_k]
  - sigma_pred^2 ≈ E[ var_k + mu_k^2 ] - (E[mu_k])^2

- For Poisson:
  ```python
  params = {
      "rate":  lambda_pred,  # E[lambda | X], shape (n, d)
      "var_y": var_y_pred    # Var(Y | X),    shape (n, d)
  }
  ```
  where `var_y_pred` accounts for both E[lambda] and Var[lambda]
  under the mixture.

For convenience, `predict_mean()` is built on top of `predict_params()`:

```python
# For Bernoulli: probabilities
# For Gaussian: predictive mean
# For Poisson:  predictive mean rate
mean_y = trainer.predict_mean(X_test, n_mc=20)
```

And for Bernoulli only, `predict_proba()` acts as a synonym for probabilities:

```python
p_hat = trainer.predict_proba(X_test, n_mc=20)  # outcome_type="bernoulli"
```

For Gaussian or Poisson, `predict_proba()` raises an error and you should use `predict_mean()` or `predict_params()`.

---

## 5.3 Sampling from the predictive distribution

The `generate()` method samples from the predictive distribution `p(Y | X)`:

```python
Y_sim = trainer.generate(
    X_new,
    n_samples_per_x=5,
    return_probs=False
)
```

Behavior by family:

- Bernoulli:
  - `return_probs=True`  → probabilities P(Y_ij = 1 | X_i)  
  - `return_probs=False` → Bernoulli samples (0/1)  
- Gaussian:
  - `return_probs=True`  → predictive means (mu_pred)  
  - `return_probs=False` → Normal draws using the decoder parameters  
- Poisson:
  - `return_probs=True`  → predictive rates (lambda_pred)  
  - `return_probs=False` → Poisson draws

You can combine these with correlation and marginal diagnostics to validate models for non-Bernoulli Y, even though `evaluate_loglik()` is currently implemented for Bernoulli only.

---

# 6. Databricks + MLflow Example (tuning + logging)

This section shows a **Databricks Python** example that:

1. Imports `multibin_cvae` from your workspace
2. Simulates Bernoulli data
3. Runs hyperparameter tuning via `fit_cvae_with_tuning()`
4. Evaluates fit
5. Logs parameters, metrics, and correlation plots to MLflow

## 6.1 Setup

```python
import sys
sys.path.append("/Workspace/Users/<user>/cvae_multibin/python")

import mlflow
mlflow.autolog(disable=True)

from multibin_cvae import (
    simulate_cvae_data,
    fit_cvae_with_tuning,
    compare_real_vs_generated,
)
```

## 6.2 Data

```python
X, Y, params_true = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234
)
```

## 6.3 Search space

```python
search_space = {
    "latent_dim":      [4, 8, 16],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40],
}
```

## 6.4 Run tuning + final fit with MLflow logging

```python
import numpy as np
import matplotlib.pyplot as plt

mlflow.set_experiment("/Users/<user>/experiments/multibin_cvae")

with mlflow.start_run(run_name="cvae_tuned_bernoulli") as run:

    outcome_type = "bernoulli"
    mlflow.log_param("outcome_type", outcome_type)
    mlflow.log_param("tuning_method", "random")
    mlflow.log_param("n_trials", 8)

    res = fit_cvae_with_tuning(
        X=X,
        Y=Y,
        search_space=search_space,
        method="random",       # or "tpe" if hyperopt is installed
        n_trials=8,
        train_frac=0.8,
        outcome_type=outcome_type,
        seed=1234,
        verbose=True,
    )

    trainer = res["trainer"]
    best_cfg = res["best_config"]
    mlflow.log_params(best_cfg)

    # Bernoulli log-likelihood metrics on a test subset
    test_idx = np.random.choice(X.shape[0], 3000, replace=False)
    metrics = trainer.evaluate_loglik(X[test_idx], Y[test_idx], n_mc=30)
    mlflow.log_metrics(metrics)

    # Correlation comparison: real vs generated on test subset
    Y_gen = trainer.generate(X[test_idx], n_samples_per_x=1, return_probs=False)
    Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

    comp = compare_real_vs_generated(
        Y_real=Y[test_idx],
        Y_gen=Y_gen_flat,
        label_real="test",
        label_gen="generated",
        make_plot=True
    )
    fig = comp.get("fig", None)
    if fig is not None:
        mlflow.log_figure(fig, "corr_test_vs_generated.png")
        plt.close(fig)

    # Log the trained PyTorch model (decoder+encoder) as an MLflow artifact
    mlflow.pytorch.log_model(trainer.model, "model")

    print("Run ID:", run.info.run_id)
```

You can later swap in **real X, Y** instead of simulated data, and/or change the outcome family (`outcome_type`) as your use-cases evolve.

