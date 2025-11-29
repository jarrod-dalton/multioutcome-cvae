# R/fit_cvae_example.R

library(reticulate)

# OPTIONAL: explicitly point to the Python env you want
# use_python("/path/to/your/python", required = TRUE)

# Make sure Python can find your multibin_cvae package.
# On Databricks, you might instead have done sys.path.append() in Python;
# here we assume the package is discoverable on sys.path.
mb <- import("multibin_cvae")

# 1. Simulate data from the Python helper
sim <- mb$simulate_cvae_data(
  n_samples  = as.integer(20000),
  n_features = as.integer(5),
  n_outcomes = as.integer(10),
  latent_dim = as.integer(2),
  seed       = as.integer(1234)
)

X <- sim[[1]]
Y <- sim[[2]]

dim(X)  # 20000 x 5
dim(Y)  # 20000 x 10

# 2. Define a small search space for tuning
# reticulate::dict() creates a Python dict
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

# 3. Run tuning + final fit (no MLflow, entirely in memory)
fit_res <- mb$fit_cvae_with_tuning(
  X           = X,
  Y           = Y,
  search_space = search_space,
  method      = "random",   # or "tpe" if hyperopt is installed
  n_trials    = as.integer(5),
  train_frac  = 0.8,
  seed        = as.integer(1234),
  verbose     = TRUE
)

# Fit results is a Python dict-like object:
fit_res
best_config <- fit_res[["best_config"]]
best_config

# 4. Extract the trained CVAETrainer
trainer <- fit_res[["trainer"]]

# 5. Generate new outcomes for a subset of X
X_new <- X[1:5, , drop = FALSE]

Y_sim <- trainer$generate(
  X_new,
  n_samples_per_x = as.integer(10),
  return_probs    = FALSE
)

# Y_sim has shape (5, 10, 10) in Python terms (R will show as array)
Y_sim

# 6. Evaluate log-likelihood-based goodness-of-fit on a held-out subset
idx_test <- 1:2000
ll_metrics <- trainer$evaluate_loglik(
  X = X[idx_test, , drop = FALSE],
  Y = Y[idx_test, , drop = FALSE],
  n_mc = as.integer(20),
  eps  = 1e-7
)

ll_metrics
