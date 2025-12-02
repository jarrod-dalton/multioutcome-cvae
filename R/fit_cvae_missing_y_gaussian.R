# R/fit_cvae_missing_y_gaussian.R
#
# Example: Fit a Gaussian CVAE with partially missing Y (outcomes) from R
# using reticulate.
#
# Assumes:
#   - Python environment has multibin_cvae installed / importable
#   - You have set up RETICULATE_PYTHON or use_python()/use_condaenv()
#
# Usage:
#   source("R/fit_cvae_missing_y_gaussian.R")

library(reticulate)

# ------------------------------------------------------------------------------
# 1. Configure Python + import multibin_cvae
# ------------------------------------------------------------------------------

# If needed, pin Python (Databricks / Posit / local):
# use_python("/databricks/python3/bin/python", required = TRUE)
# or use_condaenv("myenv", required = TRUE)

sys <- import("sys")

# EDIT this path so Python can find your package if needed.
# If you've installed from GitHub via pip, you may not need this.
# sys$path$append("/path/to/your/repo/python")

mb <- import("multibin_cvae")

# ------------------------------------------------------------------------------
# 2. Simulate complete Gaussian data
# ------------------------------------------------------------------------------

sim <- mb$simulate_cvae_data(
  n_samples  = as.integer(5000),
  n_features = as.integer(5),
  n_outcomes = as.integer(6),
  latent_dim = as.integer(2),
  outcome_type = "gaussian",
  seed       = as.integer(1234),
  noise_sd   = 1.0
)

X <- sim[[1]]
Y_full <- sim[[2]]

cat("Simulated data:\n")
cat("  dim(X)     =", paste(dim(X), collapse = " x "), "\n")
cat("  dim(Y_full)=", paste(dim(Y_full), collapse = " x "), "\n\n")

# ------------------------------------------------------------------------------
# 3. Introduce missingness in Y and build mask
# ------------------------------------------------------------------------------

set.seed(2025)
missing_prob <- 0.3

n <- nrow(Y_full)
y_dim <- ncol(Y_full)

# 1 = observed, 0 = missing
Y_mask <- matrix(
  data = as.numeric(runif(n * y_dim) > missing_prob),
  nrow = n,
  ncol = y_dim
)

Y_obs <- Y_full
Y_obs[Y_mask == 0] <- NA_real_

# Simple column-mean imputation for encoder input
col_means <- colMeans(Y_obs, na.rm = TRUE)

Y_imputed <- Y_obs
for (j in seq_len(y_dim)) {
  idx_na <- is.na(Y_imputed[, j])
  if (any(idx_na)) {
    Y_imputed[idx_na, j] <- col_means[j]
  }
}

# ------------------------------------------------------------------------------
# 4. Train/validation split
# ------------------------------------------------------------------------------

idx <- sample.int(n)
n_train <- floor(0.8 * n)

idx_train <- idx[seq_len(n_train)]
idx_val   <- idx[(n_train + 1L):n]

X_train <- X[idx_train, , drop = FALSE]
X_val   <- X[idx_val,   , drop = FALSE]

Y_train <- Y_imputed[idx_train, , drop = FALSE]
Y_val   <- Y_imputed[idx_val,   , drop = FALSE]

Y_mask_train <- Y_mask[idx_train, , drop = FALSE]
Y_mask_val   <- Y_mask[idx_val,   , drop = FALSE]

Y_full_val   <- Y_full[idx_val,   , drop = FALSE]  # for evaluation

# ------------------------------------------------------------------------------
# 5. Construct and fit the CVAETrainer (Gaussian, masked Y)
# ------------------------------------------------------------------------------

trainer <- mb$CVAETrainer(
  x_dim        = as.integer(ncol(X)),
  y_dim        = as.integer(ncol(Y_full)),
  latent_dim   = as.integer(8),
  outcome_type = "gaussian",
  hidden_dim   = as.integer(64),
  n_hidden_layers = as.integer(2)
)

history <- trainer$fit(
  X_train      = X_train,
  Y_train      = Y_train,
  X_val        = X_val,
  Y_val        = Y_val,
  Y_mask_train = Y_mask_train,
  Y_mask_val   = Y_mask_val,
  num_epochs   = as.integer(30),
  verbose      = TRUE,
  seed         = as.integer(1234)
)

cat("\nTraining history (last 5 epochs):\n")
print(tail(unlist(history$train_loss), 5))

if (length(history$val_loss) > 0) {
  cat("\nValidation history (last 5 epochs):\n")
  print(tail(unlist(history$val_loss), 5))
}

# ------------------------------------------------------------------------------
# 6. Evaluate predictive performance on observed entries in Y_val
# ------------------------------------------------------------------------------

mu_pred <- trainer$predict_mean(X_val, n_mc = as.integer(30L))

# MSE only on observed Y entries
mask_val_logical <- (Y_mask_val == 1)
diffs <- (mu_pred - Y_full_val)^2
mse_obs <- mean(diffs[mask_val_logical])

cat("\nValidation MSE on observed Y entries:", mse_obs, "\n")
