# R/use_mlflow_model.R
#
# Example: use a CVAE model stored in MLflow from R via reticulate.
#
# Prerequisites:
#   1. You ran python/multioutcome_cvae/examples/train_with_mlflow.py
#      and have an MLflow run ID.
#   2. That training script logged:
#        - the PyTorch model at artifact_path="model"
#        - x_mean.npy and x_std.npy in the run's artifact root.
#
# Usage:
#   Sys.setenv(MLFLOW_RUN_ID = "<your_run_id>")
#   source("R/use_mlflow_model.R")
#
# Notes:
#   - This example assumes outcome_type="bernoulli" and uses
#     LoadedCVAEModule defined in examples/use_mlflow_model.py.
#   - You may want to set RETICULATE_PYTHON / PATH so R and Python
#     use the same environment as your MLflow run.

library(reticulate)

# ------------------------------------------------------------------
# 1. Configure Python + import modules
# ------------------------------------------------------------------

# If needed, pin the exact Python used in Databricks / Posit:
# use_python("/databricks/python3/bin/python", required = TRUE)
# or use_condaenv("myenv", required = TRUE)

sys <- import("sys")

# Make sure Python can find the multioutcome_cvae package.
# EDIT THIS PATH for your environment:
sys$path$append("/path/to/your/repo/python")

mlflow   <- import("mlflow")
np       <- import("numpy")
torch    <- import("torch")

# Import the helper class that wraps the raw model
use_mod  <- import("multioutcome_cvae.examples.use_mlflow_model")
LoadedCVAEModule <- use_mod$LoadedCVAEModule

# ------------------------------------------------------------------
# 2. Read run ID and sanity check
# ------------------------------------------------------------------

run_id <- Sys.getenv("MLFLOW_RUN_ID")
if (run_id == "") {
  stop("Please set MLFLOW_RUN_ID env var to the run ID from train_with_mlflow.py")
}
cat("Using MLflow run ID:", run_id, "\n")

# ------------------------------------------------------------------
# 3. Load model + standardization artifacts from MLflow
# ------------------------------------------------------------------

client <- mlflow$tracking$MlflowClient()

# Download artifacts to a temporary directory
dst <- tempdir()
x_mean_path <- client$download_artifacts(run_id, "x_mean.npy", dst_path = dst)
x_std_path  <- client$download_artifacts(run_id, "x_std.npy",  dst_path = dst)

cat("Downloaded x_mean.npy to:", x_mean_path, "\n")
cat("Downloaded x_std.npy to :", x_std_path,  "\n")

x_mean <- np$load(x_mean_path)
x_std  <- np$load(x_std_path)

# Load the PyTorch model logged by train_with_mlflow.py
model_uri <- sprintf("runs:/%s/model", run_id)
model <- mlflow$pytorch$load_model(model_uri)

cat("Loaded model from:", model_uri, "\n")
cat("x_mean shape:", paste(x_mean$shape, collapse = " x "), "\n")
cat("x_std  shape:", paste(x_std$shape,  collapse = " x "), "\n\n")

# Wrap everything in the LoadedCVAEModule helper
loaded <- LoadedCVAEModule(
  model   = model,
  x_mean  = x_mean,
  x_std   = x_std,
  device  = "cpu"  # or "cuda" if GPU available and configured
)

# ------------------------------------------------------------------
# 4. Create new X in R and call the model
# ------------------------------------------------------------------

# Infer x_dim from the model (attribute was set in CVAETrainer/MultivariateOutcomeCVAE)
x_dim <- as.integer(model$x_dim)
y_dim <- as.integer(model$y_dim)

cat("Model x_dim:", x_dim, "\n")
cat("Model y_dim:", y_dim, "\n\n")

set.seed(2025)
n_new <- 5L
X_new <- matrix(rnorm(n_new * x_dim), nrow = n_new, ncol = x_dim)

# Predict probabilities P(Y_ij = 1 | X_i) via Monte Carlo
p_hat <- loaded$predict_proba(X_new, n_mc = as.integer(30L))

cat("Predicted probabilities (first new row):\n")
print(p_hat[1, ])

# Generate Bernoulli samples from the predictive distribution
Y_sim <- loaded$generate(
  X_new,
  n_samples_per_x = as.integer(5L),
  return_probs    = FALSE
)

cat("\nGenerated Y_sim has dimensions:\n")
print(dim(Y_sim))
