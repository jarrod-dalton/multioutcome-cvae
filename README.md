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

Three R scripts (using **reticulate**) live under `R/`:

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
   
- `R/fit_cvae_missing_y_gaussian.R`
  - MCAR missingness in Y
  - construction of a Y-mask matrix
  - simple imputation for encoder inputs
  - passing `Y_mask_train` and `Y_mask_val` into `CVAETrainer$fit()`
  - evaluating predictive accuracy only on observed Y entries

  This example mirrors the Python version (`gaussian_missing_y.py`) and shows how to call the masked-Y CVAE from R using **reticulate** (Databricks, Posit Workbench, or local).

These provide a template for an **R-facing front end** that uses:

- Python for CVAE training and inference  
- MLflow as a storage + registry layer  

---

## 7. Handling missing outcome values (Y)

The CVAE can be trained when **Y has missing elements**, under standard MCAR/MAR assumptions, by providing a **mask** for the observed entries.

### Mask semantics

- `Y_mask[i, j] = 1` → `Y[i, j]` is observed  
- `Y_mask[i, j] = 0` → `Y[i, j]` is missing  

During training:

- The reconstruction loss is computed **only over observed entries**.
- Missing entries do **not** contribute to the likelihood term.
- X is still assumed to be **fully observed**; handling missing X is currently out of scope.

### Training with missing Y (example, Gaussian)

```python
import numpy as np
from multibin_cvae import simulate_cvae_data, CVAETrainer

# 1. Simulate complete Gaussian data
X, Y_full, params = simulate_cvae_data(
    n_samples=5000,
    n_features=5,
    n_outcomes=6,
    latent_dim=2,
    outcome_type="gaussian",
    seed=1234,
)

# 2. Create a missingness mask for Y (30% missing completely at random)
rng = np.random.default_rng(2025)
missing_prob = 0.3
Y_mask = (rng.random(Y_full.shape) > missing_prob).astype(np.float32)  # 1=obs, 0=miss

# 3. Create an imputed version of Y for the encoder input
Y_obs = Y_full.copy()
Y_obs[Y_mask == 0] = np.nan
col_means = np.nanmean(Y_obs, axis=0)
Y_imputed = np.where(np.isnan(Y_obs), col_means, Y_obs)

# 4. Train/validation split
n = X.shape[0]
idx = rng.permutation(n)
n_train = int(0.8 * n)
idx_train, idx_val = idx[:n_train], idx[n_train:]

X_train, X_val = X[idx_train], X[idx_val]
Y_train, Y_val = Y_imputed[idx_train], Y_imputed[idx_val]
Y_mask_train, Y_mask_val = Y_mask[idx_train], Y_mask[idx_val]

# 5. Fit CVAE with masked Y
trainer = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y_full.shape[1],
    latent_dim=8,
    outcome_type="gaussian",
    hidden_dim=64,
    n_hidden_layers=2,
)

history = trainer.fit(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    Y_mask_train=Y_mask_train,
    Y_mask_val=Y_mask_val,
    num_epochs=30,
    verbose=True,
)

# 6. Use the trained model as usual
mu_pred = trainer.predict_mean(X_val, n_mc=30)
```

In this workflow:

- The CVAE uses `Y_imputed` only as an **input to the encoder**.
- The loss function uses the **mask** (`Y_mask_*`) to restrict reconstruction to observed entries.
- X must still be imputed / completed before model fitting (see discussion in the docs for missing X).

---

## 8. Databricks + MLflow

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

## 9. Model Diagnostics

## Diagnostics, calibration, and posterior predictive checks

The package includes a small diagnostics toolbox to help you assess how well
a fitted CVAE captures the conditional distribution of `Y | X`.

### 9.1. Calibration (Bernoulli / Gaussian / Poisson)

Functions:

- `calibration_curve_with_ci(y_true, y_pred, outcome_type="bernoulli", ...)`
- `plot_global_calibration(y_true, y_pred, outcome_type="bernoulli", ...)`
- `plot_per_outcome_calibration_grid(Y_true, Y_pred, outcome_type="bernoulli", ...)`
- `expected_calibration_error(y_true, y_pred, outcome_type=...)`
- `maximum_calibration_error(y_true, y_pred, outcome_type=...)`

Example (Bernoulli):

```python
from multioutcome_cvae import (
    CVAETrainer,
    simulate_cvae_data,
    plot_global_calibration,
    plot_per_outcome_calibration_grid,
    expected_calibration_error,
)

# Simulate data and fit a Bernoulli CVAE
X, Y, _ = simulate_cvae_data(
    n_samples=5000,
    n_features=5,
    n_outcomes=8,
    latent_dim=2,
    outcome_type="bernoulli",
    seed=123,
)

trainer = CVAETrainer(
    x_dim=X.shape[1],
    y_dim=Y.shape[1],
    latent_dim=4,
    outcome_type="bernoulli",
)

trainer.fit(X, Y, num_epochs=20, verbose=False, seed=123)

# Predicted probabilities on the same X (or a test set)
p_hat = trainer.predict_mean(X, n_mc=30)

# Global calibration plot (all outcomes flattened)
plot_global_calibration(
    y_true=Y.ravel(),
    y_pred=p_hat.ravel(),
    outcome_type="bernoulli",
)

# Per-outcome calibration grid
plot_per_outcome_calibration_grid(
    Y_true=Y,
    Y_pred=p_hat,
    outcome_type="bernoulli",
)

# Scalar calibration summaries
ece = expected_calibration_error(Y.ravel(), p_hat.ravel(), outcome_type="bernoulli")
print("ECE (Bernoulli):", ece)
```

For Gaussian / Poisson outcomes, pass the appropriate `outcome_type` and use
predicted means from `trainer.predict_mean(...)` in place of probabilities.

---

### 9.2. SHAP-style dependence curves

For a fitted model, you can visualize how a single feature in `X` influences
the mean of a particular outcome dimension via a simple “partial-dependence”
style curve:

Functions:

- `dependence_curve(trainer, X, feature_index, outcome_index=0, ...)`
- `plot_dependence_curve(trainer, X, feature_index, outcome_index=0, ...)`

Example:

```python
from multioutcome_cvae import plot_dependence_curve

# trainer: a fitted CVAETrainer
# X: the covariate matrix used for training or a representative sample

ax = plot_dependence_curve(
    trainer=trainer,
    X=X,
    feature_index=0,     # which X column to vary
    outcome_index=0,     # which Y dimension to track
    n_grid=50,
    n_mc=30,
    feature_name="X0",
    outcome_name="Y0",
)
```

This produces a 1D curve showing `E[Y[outcome_index] | X_feature]` as the
feature is swept over a grid of values (within a chosen quantile range).

---

### 9.3. Posterior predictive checks (Gaussian / Poisson)

These helpers compare summaries of the observed data with replicated data
drawn from the fitted CVAE’s posterior predictive distribution.

Functions:

- `posterior_predictive_check_gaussian(trainer, X, Y, n_rep=100, ...)`
- `posterior_predictive_check_poisson(trainer, X, Y, n_rep=100, ...)`

Example (Gaussian):

```python
from multioutcome_cvae import posterior_predictive_check_gaussian

# Suppose trainer_gauss is a CVAETrainer with outcome_type="gaussian"
# and X_gauss, Y_gauss are the data used for evaluation

ppc_results = posterior_predictive_check_gaussian(
    trainer=trainer_gauss,
    X=X_gauss,
    Y=Y_gauss,
    n_rep=200,
    n_mc_params=20,
    plot=True,   # show histograms of replicated vs observed summaries
)
```

Example (Poisson):

```python
from multioutcome_cvae import posterior_predictive_check_poisson

ppc_results = posterior_predictive_check_poisson(
    trainer=trainer_poiss,
    X=X_poiss,
    Y=Y_poiss,
    n_rep=200,
    n_mc_params=20,
    plot=True,
)
```

These functions return dictionaries with observed and replicated summary
statistics (means and variances across outcome dimensions), which you can
use for additional custom diagnostics.

---

### 9.4. Vignette

See the **model recovery & diagnostics vignette** (e.g.
`docs/model_recovery_vignette.ipynb` or equivalent) for a complete, runnable
example that:

- Simulates Bernoulli / Gaussian / Poisson outcomes,
- Fits a CVAE for each family,
- Computes calibration curves and ECE/MCE,
- Produces per-outcome calibration grids,
- Runs posterior predictive checks,
- And illustrates dependence curves for selected features.

---

## 10. Unit Tests

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

---

This project was developed with assistance from ChatGPT/GPT-4/5.

