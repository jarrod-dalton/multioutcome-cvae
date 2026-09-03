# multioutcome-cvae 0.3.0

Unreleased.

## Mixed-cardinality categorical outcomes

- Added engineering support for joint nominal outcomes with different numbers
  of levels, including two-level outcomes, through
  `outcome_type="categorical"` and an explicit outcome schema.
- The categorical API and deployment artifacts are versioned, but broader
  probabilistic predictive validation remains open in GitHub Issue #2; the
  family should remain experimental in downstream scientific/production use.
- Added grouped softmax likelihoods, mask-aware one-hot encoder inputs, named
  probability outputs, valid integer-code generation, and bounded expanded
  decoder batches.
- Added analytic tests showing that every two-level categorical group is
  distributionally equivalent to a Bernoulli outcome.
- Added a predeclared, generated Phase A predictive-validation workflow with
  proper marginal and joint scores, oracle comparisons, conditional-
  independence controls, quadrature diagnostics, and visible failure gates.
- The locked Phase A run is inconclusive: all marginal-improvement gates
  passed, while four joint predictive gates and 18 quadrature prerequisites
  failed. Categorical joint simulation remains experimental pending Issue #2.

## Portable inference

- Added versioned `save_cvae()` and `load_cvae()` inference checkpoints that
  retain model architecture, X scaling, outcome family, and categorical
  metadata.
- Added `export_categorical_r()` and a format-version-2 dependency-free base-R
  runtime with schema inspection, grouped probabilities, and shared-latent
  categorical simulation.
- Preserved the Bernoulli R format-version-1 exporter and its public behavior.

## Validation and documentation

- Added Python/R fixed-latent probability parity, checkpoint round-trip,
  categorical mask, batching, generation, and neutral mixed-cardinality
  recovery coverage.
- Clarified that shared latent variables model joint dependence but do not
  guarantee complete recovery of the joint distribution.
- Kept mixed discrete/continuous decoders, ordinal likelihoods, automatic label
  encoding, and standalone R training out of scope.

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
