#' Fit multivariate binary CVAE
#'
#' High-level wrapper around Python CVAETrainer.
#'
#' @param X Numeric matrix of covariates.
#' @param Y Numeric or integer matrix of 0/1 outcomes.
#' @param latent_dim,hidden_dim,num_epochs,batch_size,lr,beta_kl Hyperparameters.
#' @return A reticulate Python object (CVAETrainer).
#' @export
fit_multioutcome_cvae <- function(
  X,
  Y,
  latent_dim = 8L,
  hidden_dim = 64L,
  num_epochs = 50L,
  batch_size = 256L,
  lr = 1e-3,
  beta_kl = 1.0
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  mb <- reticulate::import("multioutcome_cvae")

  trainer <- mb$CVAETrainer(
    x_dim = ncol(X),
    y_dim = ncol(Y),
    latent_dim = as.integer(latent_dim),
    hidden_dim = as.integer(hidden_dim),
    num_epochs = as.integer(num_epochs),
    batch_size = as.integer(batch_size),
    lr = lr,
    beta_kl = beta_kl
  )

  trainer$fit(
    X_train = X,
    Y_train = Y,
    X_val = NULL,
    Y_val = NULL,
    verbose = TRUE
  )

  trainer
}
