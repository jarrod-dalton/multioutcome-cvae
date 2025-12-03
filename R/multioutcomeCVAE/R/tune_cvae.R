#' Tune multivariate binary CVAE via random search
#'
#' @param X_train,Y_train,X_val,Y_val Matrices of train/validation data.
#' @param search_space Named list of candidate values (each element a vector).
#' @param n_trials Number of random search trials.
#' @return A list containing trial results and best config.
#' @export
tune_multioutcome_cvae <- function(
  X_train,
  Y_train,
  X_val,
  Y_val,
  search_space,
  n_trials = 10L
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
  X_train <- as.matrix(X_train)
  Y_train <- as.matrix(Y_train)
  X_val   <- as.matrix(X_val)
  Y_val   <- as.matrix(Y_val)

  mb <- reticulate::import("multioutcome_cvae")

  # Convert R list to Python dict
  py_search_space <- reticulate::r_to_py(search_space)

  res <- mb$tune_cvae_random_search(
    X_train = X_train,
    Y_train = Y_train,
    X_val   = X_val,
    Y_val   = Y_val,
    x_dim   = ncol(X_train),
    y_dim   = ncol(Y_train),
    search_space = py_search_space,
    n_trials = as.integer(n_trials)
  )
  res
}
