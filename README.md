# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate outcomes**.

Goal: estimate and simulate from flexible conditional distributions

- **X** = covariates / predictors  
- **Y** = multivariate outcome (binary, continuous, or counts; same family per repo run)

so you can:

1. Estimate **p(Y | X)** (or E[Y | X])  
2. Generate **realistic, correlated Y vectors** for new X  
3. Evaluate fit using log-likelihood (Bernoulli) and correlation diagnostics  

Includes:

- Flexible CVAE model with selectable outcome family  
- Random search and TPE hyperparameter tuning  
- Log-likelihood evaluation (Bernoulli)  
- Correlation/marginal diagnostic heatmaps  
- Databricks + MLflow training example  
- R integration via `reticulate`  
- Convenience `fit_cvae_with_tuning()` wrapper  

---

# Table of Contents

- [1. Repository Structure](#1-repository-structure)  
- [2. Installation / Environment](#2-installation--environment)  
- [3. Basic Usage Examples (Python, R, Databricks)](#3-basic-usage-examples-python-r-databricks)  
- [4. Evaluation Notes](#4-evaluation-notes)  
- [5. Multiple Outcome Families for Y](#5-multiple-outcome-families-for-y)  
- [6. Databricks + MLflow Example (with tuning)](#6-databricks--mlflow-example-with-tuning)  

---

# 1. Repository Structure

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py
│       ├── model.py             # CVAETrainer, tuning, fit_cvae_with_tuning()
│       ├── simulate.py          # data simulation + correlation utilities
│       └── examples             # (optional, for local scripts)
├── R
│   └── fit_cvae_example.R       # R/reticulate usage example
└── README.md
```

Key modules:

- `model.py` – CVAE model + trainer + tuning utilities  
- `simulate.py` – simulated data + summary + correlation heatmaps  

---

# 2. Installation / Environment

The code is *source-only* (not on PyPI). You can use it from:

- Python (local/server)
- RStudio / Posit Workbench via **reticulate**
- Databricks (Python and R notebooks)

### 2.1 Python dependencies

Core:

- `numpy`
- `torch`
- `matplotlib`

Optional:

- `mlflow`   – experiment tracking (Databricks-native)
- `hyperopt` – TPE tuning

Example install:

```bash
pip install numpy torch matplotlib mlflow hyperopt
```

### 2.2 Making `multibin_cvae` importable

#### Option A (quick)

```python
import sys
sys.path.append("/path/to/repo/python")

import multibin_cvae
```

#### Option B (later)

Add `pyproject.toml` or `setup.cfg` and:

```bash
pip install -e python/
```

---

# 3. Basic Usage Examples (Python, R, Databricks)

## 3.1 Python (Bernoulli outcomes)

```python
import sys
sys.path.append("/path/to/repo/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

# simulate binary outcomes
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
    outcome_type="bernoulli",      # default; explicit for clarity
    enc_hidden_dims=[64, 64],
    dec_hidden_dims=[64, 64],
)

trainer.fit(X, Y, verbose=True)

# generate new binary outcomes
Y_sim = trainer.generate(X[:10], n_samples_per_x=5)
```

---

## 3.2 Python (Gaussian or Poisson outcomes)

```python
from multibin_cvae import CVAETrainer

# Suppose Y is continuous:
#   outcome_type="gaussian"
trainer_g = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=8,
    outcome_type="gaussian"
)
trainer_g.fit(X, Y)

# E[Y | X] (means)
Y_mean = trainer_g.predict_mean(X[:10])

# Gaussian samples
Y_gauss = trainer_g.generate(X[:10], n_samples_per_x=5, return_probs=False)
```

```python
# Suppose Y is counts:
trainer_p = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=8,
    outcome_type="poisson"
)
trainer_p.fit(X, Y)

# Rates lambda(X)
lambda_hat = trainer_p.predict_mean(X[:10])

# Poisson samples
Y_pois = trainer_p.generate(X[:10], n_samples_per_x=5, return_probs=False)
```

---

## 3.3 RStudio / Posit via reticulate (Bernoulli)

```r
library(reticulate)

# Optional: pin environment
# use_python("/opt/miniconda/envs/cvae/bin/python", required = TRUE)

mb <- import("multibin_cvae")

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
  x_dim       = ncol(X),
  y_dim       = ncol(Y),
  latent_dim  = 8L,
  outcome_type = "bernoulli"
)

trainer$fit(X, Y)

Y_sim <- trainer$generate(X[1:5,], n_samples_per_x = 10L)
```

For Gaussian/Poisson, change `outcome_type` accordingly and use `predict_mean()` for E[Y | X].

---

## 3.4 Databricks (Python notebook, Bernoulli)

```python
import sys
sys.path.append("/Workspace/Users/<user>/cvae_multibin/python")

from multibin_cvae import simulate_cvae_data, CVAETrainer

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

trainer.fit(X, Y)
Y_sim = trainer.generate(X[:5], n_samples_per_x=5)
```

---

# 4. Evaluation Notes

Evaluation combines:

- a **log-likelihood / cross-entropy style measure** (Bernoulli only)
- **correlation + marginal summaries** (any outcome family)

## 4.1 Log-likelihood / cross-entropy (Bernoulli)

For Bernoulli Y, with:

- Y: n × d matrix of 0/1 outcomes  
- p_hat[i,j]: posterior-averaged probability E_z[p(Y[i,j]=1 | X[i], z)]

we define:

```text
LL = sum_{i=1..n} sum_{j=1..d} [
        Y[i,j]     * log(p_hat[i,j]) +
        (1 - Y[i,j]) * log(1 - p_hat[i,j])
    ]

avg_LL  = LL / (n * d)
avg_BCE = -avg_LL
```

Computed via:

```python
metrics = trainer.evaluate_loglik(X_test, Y_test)
# Only valid for outcome_type="bernoulli"
```

For non-Bernoulli outcomes, `evaluate_loglik()` currently raises `NotImplementedError`.

---

## 4.2 Correlation and marginal structure

Regardless of outcome family, you can inspect marginals and correlations:

```python
from multibin_cvae import summarize_binary_matrix, compare_real_vs_generated

# For any numeric Y (binary, counts, continuous)
summary_real = summarize_binary_matrix(Y_real, name="real", make_plot=True)

Y_gen = trainer.generate(X_real, n_samples_per_x=1)
Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])

summary = compare_real_vs_generated(
    Y_real=Y_real,
    Y_gen=Y_gen_flat,
    label_real="real",
    label_gen="generated",
    make_plot=True
)
```

Plots:

- marginal prevalence/means  
- side-by-side correlation heatmaps (diagonal masked)  

These are especially useful for checking **joint dependence structure**.

---

# 5. Multiple Outcome Families for Y

The repo now supports three outcome families, applied uniformly across the components of Y:

- `"bernoulli"` – binary outcomes (0/1)  
- `"gaussian"`  – continuous outcomes  
- `"poisson"`   – count outcomes  

### 5.1 Selecting outcome_type

In all high-level APIs:

- `CVAETrainer(...)`
- `tune_cvae_random_search(...)`
- `tune_cvae_tpe(...)`
- `fit_cvae_with_tuning(...)`

you can set:

```python
outcome_type="bernoulli"  # default
outcome_type="gaussian"
outcome_type="poisson"
```

### 5.2 Decoder parameterization

Internally, the decoder uses:

- Bernoulli: logits  
- Gaussian: mean `mu` and log-variance `logvar`  
- Poisson: log-rate `log_rate`

Reconstruction losses are the negative log-likelihoods (up to additive constants):

- Bernoulli: binary cross-entropy  
- Gaussian: 0.5 * [ logvar + (y-mu)^2 / exp(logvar) ]  
- Poisson:  rate - y * log_rate  

### 5.3 Prediction helpers

- `predict_mean(X, n_mc)`  
  - Bernoulli: returns probabilities  
  - Gaussian: returns means  
  - Poisson:  returns rates λ  

- `predict_proba(X, n_mc)`  
  - **Only** defined for `outcome_type="bernoulli"`  
  - For Gaussian/Poisson, it raises an error; use `predict_mean()` instead.

- `generate(X, n_samples_per_x, return_probs)`  
  - Bernoulli:
    - `return_probs=True` → probabilities (via MC mean)  
    - `return_probs=False` → Bernoulli samples  
  - Gaussian:
    - `return_probs=True` → means  
    - `return_probs=False` → Normal samples  
  - Poisson:
    - `return_probs=True` → rates λ  
    - `return_probs=False` → Poisson samples  

This keeps Bernoulli behavior backwards-compatible while generalizing to other Y families.

---

# 6. Databricks + MLflow Example (with tuning)

This is a self-contained Databricks Python notebook sketch that:

1. Imports the repo  
2. Simulates (or loads) data  
3. Runs hyperparameter tuning via `fit_cvae_with_tuning()`  
4. Logs metrics and artifacts to MLflow  

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
X, Y, params = simulate_cvae_data(
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
        method="random",
        n_trials=8,
        train_frac=0.8,
        outcome_type=outcome_type,
        seed=1234,
        verbose=True,
    )

    trainer = res["trainer"]
    best_cfg = res["best_config"]
    mlflow.log_params(best_cfg)

    # log-likelihood metrics (Bernoulli only)
    test_idx = np.random.choice(X.shape[0], 3000, replace=False)
    metrics = trainer.evaluate_loglik(X[test_idx], Y[test_idx])
    mlflow.log_metrics(metrics)

    # correlation heatmap for test vs generated
    Y_gen = trainer.generate(X[test_idx], n_samples_per_x=1)
    Y_gen = Y_gen.reshape(-1, Y_gen.shape[-1])

    comp = compare_real_vs_generated(
        Y_real=Y[test_idx],
        Y_gen=Y_gen,
        label_real="test",
        label_gen="generated",
        make_plot=True
    )
    fig = comp.get("fig", None)
    if fig is not None:
        mlflow.log_figure(fig, "corr_test_vs_generated.png")
        plt.close(fig)

    # save model only (preprocessing can be stored separately)
    mlflow.pytorch.log_model(trainer.model, "model")

    print("Run ID:", run.info.run_id)
```

You can swap in real X, Y (e.g., from Snowflake / Delta tables) and choose a different `outcome_type` as needed. For non-Bernoulli families, you’d typically log:

- `predict_mean`-based metrics (MSE, Poisson deviance, etc.)  
- correlation heatmaps and marginals rather than the Bernoulli log-likelihood.
