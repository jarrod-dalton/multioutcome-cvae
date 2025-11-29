# multibin-cvae

Tools for fitting and using a **Conditional Variational Autoencoder (CVAE)** for **multivariate binary outcomes**.

Goal: learn a flexible model for

- X = covariates / predictors  
- Y = vector of binary outcomes (e.g., 10 correlated Bernoulli variables)

so that you can:

1. Estimate **p(Y | X)** (via conditional probabilities for each Y component).  
2. Generate **realistic, correlated Y vectors** for new X.

Includes:

- Data simulation helpers
- Flexible CVAE architecture (configurable hidden layers)
- Hyperparameter tuning:
  - Random search
  - TPE (Tree-structured Parzen Estimator) via `hyperopt`
- Log-likelihood–based evaluation
- Correlation heatmaps
- MLflow logging example
- R integration via `reticulate`
- A convenience function `fit_cvae_with_tuning()` for users *not* on MLflow

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

You can adapt this layout to your own setup; the critical pieces are `model.py` and `simulate.py`.

---

## Installation / environment

You can use this code from:

- Native Python (local or server)
- RStudio / Posit Workbench via `reticulate`
- Databricks (Python and R notebooks)

The package is currently "source-only" (not on PyPI). Simplest pattern:

- Make sure `python/` is on `PYTHONPATH` (or add it via `sys.path.append()`), **or**
- Add packaging metadata later (`pyproject.toml` or `setup.py`) and `pip install -e python/`.

### Python dependencies

Core:

- `numpy`
- `torch`
- `matplotlib`

Optional:

- `mlflow` (only if you use MLflow tracking)
- `hyperopt` (only if you use TPE tuning)

Example (local Python):

```bash
pip install numpy torch matplotlib mlflow hyperopt
```

---

## 1. Native Python usage

In a regular Python environment:

```python
import sys
sys.path.append("/path/to/your/repo/python")  # so 'multibin_cvae' is importable

from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    CVAETrainer,
    fit_cvae_with_tuning,
    compare_real_vs_generated,
)

# Simulate data
X, Y, params_true = simulate_cvae_data(
    n_samples=20000,
    n_features=5,
    n_outcomes=10,
    latent_dim=2,
    seed=1234,
)

# Split into train/val/test
splits = train_val_test_split(X, Y, train_frac=0.7, val_frac=0.15, seed=1234)
X_train, Y_train = splits["X_train"], splits["Y_train"]
X_val,   Y_val   = splits["X_val"],   splits["Y_val"]
X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

# Define search space for tuning (discrete candidate sets)
search_space = {
    "latent_dim":      [4, 8, 16],
    "enc_hidden_dims": [[64, 64], [128, 64]],
    "dec_hidden_dims": [[64, 64], [128, 64]],
    "lr":              [1e-3, 5e-4],
    "beta_kl":         [0.5, 1.0],
    "batch_size":      [128, 256],
    "num_epochs":      [20, 40],
}

# Tune + fit on full data (no MLflow required)
fit_res = fit_cvae_with_tuning(
    X=X,
    Y=Y,
    search_space=search_space,
    method="random",   # or "tpe" if hyperopt is installed
    n_trials=10,
    train_frac=0.8,
    seed=1234,
    verbose=True,
)

trainer = fit_res["trainer"]

# Generate new Y for some X
X_new = X_test[:5]
Y_gen = trainer.generate(X_new, n_samples_per_x=10, return_probs=False)

# Compare correlation structure between real and generated
Y_gen_flat = Y_gen.reshape(-1, Y_gen.shape[-1])
compare_real_vs_generated(Y_real=Y_test, Y_gen=Y_gen_flat, make_plot=True)
```

Key Python API pieces:

- `CVAETrainer`:
  - `.fit(X_train, Y_train, X_val, Y_val, ...)`
  - `.generate(X_new, n_samples_per_x, return_probs)`
  - `.predict_proba(X, n_mc)` – estimated conditional probabilities
  - `.evaluate_loglik(X, Y, ...)` – log-likelihood-based GOF
- `fit_cvae_with_tuning(X, Y, search_space, ...)`:
  - Performs train/val split
  - Hyperparameter tuning (random or TPE)
  - Refit on the full dataset with best config
  - Returns a dict with `trainer`, `best_config`, `tuning_results`, `train_indices`, `val_indices`

---

## 2. RStudio / Posit Workbench usage (via reticulate)

You’ll need:

- A Python environment with `multibin_cvae` importable.
- R packages: `reticulate` (and whatever else you like).

Minimal example (see `R/fit_cvae_example.R`):

```r
library(reticulate)

# Optional: pin specific Python
# use_python("/path/to/python", required = TRUE)

# Import the Python package
mb <- import("multibin_cvae")

# 1. Simulate data
sim <- mb$simulate_cvae_data(
  n_samples  = as.integer(20000),
  n_features = as.integer(5),
  n_outcomes = as.integer(10),
  latent_dim = as.integer(2),
  seed       = as.integer(1234)
)

X <- sim[[1]]
Y <- sim[[2]]

# 2. Define search space as a Python dict
search_space <- dict(
  latent_dim      = list(as.integer(4), as.integer(8)),
  enc_hidden_dims = list(
    list(as.integer(64), as.integer(64)),
    list(as.integer(128), as.integer(64))
  ),
  dec_hidden_dims = list(
    list(as.integer(64), as.integer(64)),
    list(as.integer(128), as.integer(64))
  ),
  lr              = list(1e-3, 5e-4),
  beta_kl         = list(0.5, 1.0),
  batch_size      = list(as.integer(128), as.integer(256)),
  num_epochs      = list(as.integer(20), as.integer(40))
)

# 3. Tune + fit on full data
fit_res <- mb$fit_cvae_with_tuning(
  X            = X,
  Y            = Y,
  search_space = search_space,
  method       = "random",  # or "tpe" if hyperopt is installed
  n_trials     = as.integer(5),
  train_frac   = 0.8,
  seed         = as.integer(1234),
  verbose      = TRUE
)

trainer <- fit_res[["trainer"]]
best_config <- fit_res[["best_config"]]

# 4. Generate new outcomes for a subset of X
X_new <- X[1:5, , drop = FALSE]
Y_sim <- trainer$generate(
  X_new,
  n_samples_per_x = as.integer(10),
  return_probs    = FALSE
)

# 5. Log-likelihood-style goodness-of-fit on held-out subset
ll_metrics <- trainer$evaluate_loglik(
  X    = X[1:2000, , drop = FALSE],
  Y    = Y[1:2000, , drop = FALSE],
  n_mc = as.integer(20),
  eps  = 1e-7
)

print(ll_metrics)
```

Tips:

- Use `py_config()` to verify which Python `reticulate` is using.
- Make sure that Python’s `sys.path` includes your `python/` directory (or the package is installed/editable).

---

## 3. Databricks usage

### Python notebooks

1. Attach a cluster with `torch`, `numpy`, `matplotlib`, and (optionally) `mlflow`, `hyperopt`.
2. Ensure your repo is available under `/Workspace/...` and add the path:

```python
import sys
sys.path.append("/Workspace/Users/your.name@org.org/cvae_multibin/python")

from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    CVAETrainer,
    fit_cvae_with_tuning,
)
```

3. Use either:

- `fit_cvae_with_tuning()` for quick experiments, or
- the MLflow example in `python/multibin_cvae/examples/train_with_mlflow.py` if you want experiment tracking and artifact logging.

### R notebooks on Databricks

- Use `reticulate::py_config()` to confirm the Python env.
- You may want an initial `%python` cell to do:

```python
import sys
sys.path.append("/Workspace/Users/your.name@org.org/cvae_multibin/python")
```

- Then in R, `mb <- import("multibin_cvae")` as in the R example above.

---

## 4. MLflow example

The file:

```text
python/multibin_cvae/examples/train_with_mlflow.py
```

shows a full MLflow workflow:

- Simulate data
- Train a CVAE with chosen hyperparameters
- Log:
  - Experiment parameters
  - Final train/validation losses
  - Log-likelihood-based metrics on a test set (`evaluate_loglik`)
  - Correlation heatmap comparing real vs generated test Y (`mlflow.log_figure`)
  - The trained PyTorch model via `mlflow.pytorch.log_model`
  - Standardization parameters (`x_mean`, `x_std`) as a small artifact

You can adapt that file to load **real** data (from Snowflake, Postgres, Delta tables, etc.) instead of simulation.

---

## 5. Evaluation notes

There are two complementary evaluation perspectives:

1. **Log-likelihood / cross-entropy** (per element):

   The function `CVAETrainer.evaluate_loglik(X, Y, ...)` computes an MC-approximation of

   \[
   \sum_{i,j} \left[ y_{ij} \log(p_{ij}) + (1 - y_{ij}) \log(1 - p_{ij}) \right],
   \]

   where \(p_{ij} \approx \mathbb{E}_z[p(Y_{ij}=1 \mid X_i, z)]\).

   - Returns:
     - `sum_loglik`
     - `avg_loglik` (per i,j)
     - `avg_bce` (negative `avg_loglik`)

2. **Correlation structure**:

   - `summarize_binary_matrix(Y, make_plot=TRUE)`:
     - prints marginal probabilities
     - returns correlation matrix
     - optionally plots a heatmap (diagonal masked)
   - `compare_real_vs_generated(Y_real, Y_gen, make_plot=TRUE)`:
     - compares marginals
     - plots side-by-side correlation heatmaps
     - useful for checking whether the CVAE captures the multivariate dependence structure, not just marginals.

---

## 6. Future directions

Potential future enhancements:

- Add packaging metadata (e.g., `pyproject.toml`) for cleaner installs.
- Add calibration plots, Brier scores, or other scoring rules based on `predict_proba()`.
- Integrate with Databricks Model Registry on top of the MLflow example.
- Add a small Shiny or Dash app to interactively explore correlation structures and conditional distributions implied by a fitted CVAE.

For now, the core functionality is:

- Simulate or load real (X, Y)
- Tune + fit a CVAE (`fit_cvae_with_tuning` or MLflow path)
- Generate correlated binary outcome vectors for new covariates
- Evaluate and visualize how well the model matches the true structure.
