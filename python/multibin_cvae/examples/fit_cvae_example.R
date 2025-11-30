# R/fit_cvae_example.R
#
# Minimal R/reticulate example for multibin_cvae:
#
#   1. Import the Python package
#   2. Simulate Bernoulli data (X, Y)
#   3. Fit a CVAETrainer
#   4. Get probabilities and simulated outcomes
#
# You can source() this in RStudio / Posit Workbench, or copy the
# code into a notebook chunk.

library(reticulate)

# If needed, pin an environment:
# use_python("/databricks/python3/bin/python", required = TRUE)
# or use_condaenv("myenv", required = TRUE)

# Make sure Python can see /path/to/your/repo/python
# For Databricks or Posit, you might set this via RETICULATE_PYTHON
# and PYTHONPATH instead.
sys <- import("sys")
sys$path$append("/path/to/your/repo/python")

mb <- import("multibin_cvae")

# 1. Simulate Bernoulli data
sim <- mb$simulate_cvae_data(
  n_samples  = 5000L,
  n_features = 5L,
  n_outcomes = 10L,
  latent_dim = 2L,
  outcome_type = "bernoulli",
  seed       = 1234L
)

X <- sim[[1]]
Y <- sim[[2]]

cat("Simulated data:\n")
cat("  dim(X) =", paste(dim(X), collapse = " x "), "\n")
cat("  dim(Y) =", paste(dim(Y), collapse = " x "), "\n\n")

# 2. Construct trainer
trainer <- mb$CVAETrainer(
  x_dim        = as.integer(ncol(X)),
  y_dim        = as.integer(ncol(Y)),
  latent_dim   = 8L,
  outcome_type = "bernoulli",
  hidden_dim   = 64L,
  n_hidden_layers = 2L
)

# 3. Fit
history <- trainer$fit(X, Y, verbose = TRUE)

# 4. Predict conditional probabilities P(Y_ij = 1 | X_i)
X_sub <- X[1:10, , drop = FALSE]
p_hat <- trainer$predict_proba(X_sub, n_mc = 30L)

cat("First row of predicted probabilities:\n")
print(p_hat[1, ])

# 5. Generate new Bernoulli outcomes for these X values
Y_sim <- trainer$generate(
  X_sub,
  n_samples_per_x = 5L,
  return_probs    = FALSE
)

cat("\nGenerated Y_sim shape (flattened):\n")
print(dim(Y_sim))
