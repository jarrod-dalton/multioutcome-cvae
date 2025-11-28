#' Simulate data from the CVAE benchmark generator (Python)
#'
#' @param n_samples,n_features,n_outcomes,latent_dim,seed Arguments passed to Python simulate_cvae_data().
#' @return A list with X, Y, params.
#' @export
simulate_multibin_cvae <- function(
  n_samples = 20000L,
  n_features = 5L,
  n_outcomes = 10L,
  latent_dim = 2L,
  seed = 1234L
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }
  mb <- reticulate::import("multibin_cvae")
  sim <- mb$simulate_cvae_data(
    n_samples = as.integer(n_samples),
    n_features = as.integer(n_features),
    n_outcomes = as.integer(n_outcomes),
    latent_dim = as.integer(latent_dim),
    seed = as.integer(seed)
  )
  list(
    X = sim[[1]],
    Y = sim[[2]],
    params = sim[[3]]
  )
}
