# multibin-cvae

A small, reusable library for fitting **Conditional Variational Autoencoders (CVAEs)** for **multivariate outcomes**:

- **X** = covariates (R^p)
- **Y** = multivariate outcome (R^d), typically:
  - multivariate **Bernoulli** (0/1)
  - multivariate **Gaussian** (continuous)
  - multivariate **Poisson** (counts)

The CVAE learns a low-dimensional **latent representation Z** such that:

> Y ⟂⟂ Y' | (X, Z)

and then uses that latent structure to:

- estimate **p(Y | X)**  
- generate **correlated outcome vectors** for new X  

Most users will interact with:

- `simulate_cvae_data(...)` – convenience simulators
- `CVAETrainer` – train/evaluate/generate
- the **example scripts** under `python/multibin_cvae/examples/`
- the **R scripts** under `R/`

---

## 1. Installation

You can install directly from GitHub:

```bash
pip install "git+https://github.com/jarrod-dalton/multioutcome-cvae.git"
```

---

## 2. Quick start (Python, Bernoulli)

Minimal end-to-end example using **simulated multivariate Bernoulli** data:

```python
import sys
sys.path.append("/path/to/your/repo/python")

from multibin_cvae import (
    simulate_cvae_data,
    CVAETrainer,
)

# 1. Simulate multivariate Bernoulli Y
X, Y, params = simulate_cvae_data(
    n_samples=5000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    outcome_type="bernoulli",
    seed=1234,
)

# 2. Fit a CVAE
trainer = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=8,
    outcome_type="bernoulli",
    hidden_dim=64,
    n_hidden_layers=2,
)

trainer.fit(X, Y, verbose=True)

# 3. Predict probabilities P(Y_ij = 1 | X_i)
p_hat = trainer.predict_proba(X[:10], n_mc=30)

# 4. Generate new Bernoulli outcomes given X
Y_sim = trainer.generate(
    X[:5],
    n_samples_per_x=10,
    return_probs=False,
)
```

For more detailed examples (plots, tuning, MLflow, etc.), see the scripts under `python/multibin_cvae/examples/` summarized below.

---

## 3. Repository layout

```text
.
├── python
│   └── multibin_cvae
│       ├── __init__.py
│       ├── model.py        # CVAE model + trainer + tuning utilities
│       ├── simulate.py     # core simulators + basic diagnostics
│       └── examples        # runnable scripts (see Section 3)
└── R
    ├── fit_cvae_example.R  # basic R/reticulate usage
    └── use_mlflow_model.R  # R consumption of MLflow-stored model
```

---

## 4. Python examples

All examples are runnable as:

```bash
python -m multibin_cvae.examples.<script_name>
```

from within `python/`’s parent directory (or any location where `multibin_cvae` is importable).

### 4.1 Hello world examples

**Multivariate Bernoulli**:

- `examples/bernoulli_basic.py`  
  - Simulate multivariate Bernoulli data (`simulate_cvae_data(..., outcome_type="bernoulli")`)
  - Fit a CVAE (`CVAETrainer`)
  - Evaluate a log-likelihood-style metric on a test subset
  - Compare real vs generated correlation structure

**Multivariate Gaussian**:

- `examples/gaussian_basic.py`  
  - Simulate Gaussian outcomes (`outcome_type="gaussian"`)
  - Fit a Gaussian CVAE
  - Use `predict_params()` to examine predictive means / SD
  - Generate samples and compare correlation structure

**Multivariate Poisson**:

- `examples/poisson_basic.py`  
  - Simulate Poisson outcomes (`outcome_type="poisson"`)
  - Fit a Poisson CVAE
  - Use `predict_params()` to examine predictive rate / Var(Y | X)
  - Generate samples and compare correlation structure

These scripts are intended as small, readable “starting points” rather than production code.

---

### 4.2 MLflow training + tuning

**Train and log a model to MLflow**:

- `examples/train_with_mlflow.py`  
  - Simulates Bernoulli data
  - Runs hyperparameter tuning (`fit_cvae_with_tuning()`)
  - Refits the best configuration on the full dataset
  - Evaluates log-likelihood metrics on a test subset
  - Compares real vs generated correlation heatmaps
  - Logs:
    - model (`mlflow.pytorch.log_model`)
    - standardization parameters (`x_mean.npy`, `x_std.npy`)
    - metrics and plots

You get back an MLflow run ID that can be used for downstream inference.

---

### 4.3 Loading a model from MLflow (Python)

**Use a model stored in MLflow for new predictions / simulations**:

- `examples/use_mlflow_model.py`  
  - Reads `MLFLOW_RUN_ID` from the environment
  - Loads:
    - the logged PyTorch model
    - `x_mean.npy` and `x_std.npy` artifacts
  - Wraps them in a small `LoadedCVAEModule` helper
  - Simulates new X, then:
    - calls `predict_proba(X_new, n_mc=...)`
    - calls `generate(X_new, n_samples_per_x=...)`

This is the “Python consumer” side of the MLflow run created by `train_with_mlflow.py`.

---

## 5. Outcome families

The core `CVAETrainer` supports three outcome types:

- `outcome_type="bernoulli"` – multivariate 0/1 outcomes  
- `outcome_type="gaussian"`  – multivariate continuous outcomes  
- `outcome_type="poisson"`   – multivariate count outcomes  

Key methods:

- `fit(X, Y, ...)` – train the CVAE
- `predict_params(X, n_mc=...)` – return predictive distribution parameters:
  - Bernoulli: `{"probs": p_ij}`
  - Gaussian: `{"mu": mu_ij, "sigma": sigma_ij}`
  - Poisson:  `{"rate": lambda_ij, "var_y": Var(Y_ij | X_i)}`
- `predict_mean(X, n_mc=...)` – returns the expected value E[Y | X]
- `generate(X, n_samples_per_x, return_probs=...)` – draw samples from p(Y | X)

For all three families, the cross-dimension dependence in Y is induced through the shared latent space Z.

---

## 6. R integration

Two R scripts (using **reticulate**) live under `R/`:

- `R/fit_cvae_example.R`  
  - Imports `multibin_cvae` from Python
  - Simulates Bernoulli data in Python
  - Fits a CVAE
  - Calls `predict_proba()` and `generate()` from R

- `R/use_mlflow_model.R`  
  - Reads an MLflow run ID from `MLFLOW_RUN_ID`
  - Uses Python `mlflow` to:
    - load the logged PyTorch model
    - download `x_mean.npy` and `x_std.npy`
  - Wraps the model with the same `LoadedCVAEModule` used in `use_mlflow_model.py`
  - Simulates new covariates X in R and obtains:
    - predictive probabilities
    - simulated outcomes

These provide a template for an **R-facing front end** that uses:

- Python for CVAE training and inference  
- MLflow as a storage + registry layer  

---

## 7. Databricks + MLflow

On Databricks, the typical workflow is:

1. Clone or mount this repo into your workspace.
2. Add `/Workspace/Users/<you>/cvae_multibin/python` (or similar) to `sys.path`.
3. Use:
   - the basic examples (`bernoulli_basic.py`, etc.) in Python notebooks for quick validation
   - `train_with_mlflow.py` to run tuning + model logging
4. From R notebooks (with `reticulate`), use:
   - `fit_cvae_example.R` for direct training in a shared environment
   - `use_mlflow_model.R` to consume MLflow-stored models

---

## 8. Tests

This repository includes a small `/tests` directory with PyTest-based unit tests covering:

- simulator behavior (Bernoulli / Gaussian / Poisson)
- log-likelihood utilities
- CVAETrainer forward pass, training, prediction, and generation
- end-to-end “smoke tests” for tiny models

Tests can be run locally with:

```bash
pytest -q
```

On Databricks, you can run:

```python
%sh
pip install pytest
pytest /Workspace/Users/<you>/cvae_multibin/tests -q
```

The tests are designed to be stable across platforms and catch API regressions without over-constraining stochastic model behavior.

---

For concrete, runnable code, see the scripts in:

- `python/multibin_cvae/examples/`
- `R/`

They are intended as living documentation of how to wire everything together in real workflows.
