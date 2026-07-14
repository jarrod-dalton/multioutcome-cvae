# multioutcome-cvae 0.2.0

Released 2026-07-14.

## New features

- Added `export_bernoulli_r()` to generate dependency-free base R source from a fitted Bernoulli CVAE.
- Added base R functions for fixed-latent decoder probabilities, Monte Carlo marginal probabilities, and shared-latent conditional Bernoulli simulation.
- Added configurable row chunking for bounded-memory Python prediction and generation.
- Added KL warm-up, gradient clipping, deterministic validation, early stopping, and best-checkpoint restoration.
- Added per-latent KL and active-latent-unit training diagnostics.
- Added pairwise joint-probability and residual-covariance diagnostics for Bernoulli dependence recovery.

## Validation and correctness

- Added explicit shape, dimension, finite-value, and Bernoulli outcome validation.
- Prevented prediction data from silently fitting the X standardizer.
- Corrected epoch metrics to weight batches by their actual row counts.
- Split reconstruction, KL, and per-outcome reconstruction histories.
- Renamed the Bernoulli evaluation method to `evaluate_marginal_log_score()` to clarify that it is not a joint likelihood. The previous `evaluate_loglik()` name remains as a deprecated alias.
- Added executable Python/R parity tests using fixed covariates and latent values.
- Added seeded R simulation and decoder batch-bound regression tests.

## Support policy

- Bernoulli and Gaussian are the production-supported outcome families.
- Poisson remains available for compatibility but now emits a deprecation warning and has no production support guarantee.
- Negative Binomial remains unavailable through the public trainer API.
- Production fitting now requires complete, finite X and Y; imputation must be performed upstream.

## Packaging

- Reduced core dependencies to NumPy and PyTorch.
- Moved plotting, Hyperopt tuning, MLflow, and testing tools to optional dependency groups.
- Consolidated package metadata in `pyproject.toml` and removed the conflicting legacy `python/setup.py`.
- Added an R-enabled CI job for standalone export parity.