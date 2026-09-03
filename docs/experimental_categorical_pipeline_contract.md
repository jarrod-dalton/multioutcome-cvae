# Experimental categorical pipeline contract

The mixed-cardinality categorical API and its inference artifacts are suitable
for integration testing. They are not evidence that a fitted CVAE has adequate
probabilistic predictive validity. That evidence is tracked separately in
[GitHub Issue #2](https://github.com/jarrod-dalton/multioutcome-cvae/issues/2).
Until that work passes its predeclared gates, downstream code must keep this
model replaceable and must not describe its output as production-validated.

## Current evidence gate

The locked [Phase A joint-distribution report](categorical_predictive_validation.md)
is **inconclusive**. The complete
[conditional-probability report](categorical_probability_validation.md) is
more directly adverse: 0/34 regimes met the predeclared cell envelope, no cell
met all five aggregate predictive checks even before the every-seed rule was
applied, and only 1/170 base fits met both predictive and numerical criteria.
The correctly specified independent-softmax comparator was more accurate in
157/170 matched focal-MAE comparisons and 168/170 equal-outcome total-variation
comparisons. More training data did not reliably improve the frozen CVAE, and
large-n fits also developed material GH21-versus-GH31 integration disagreement.

This is not a failure of the input/output API. It is evidence that the present
training and marginal-integration workflow is not ready to supply calibrated
conditional probabilities or learned residual dependence for production or
scientific decisions. Ranking measures can still appear reasonable while
individual probability estimates are too compressed toward the population
mean.

It is reasonable to build reversible plumbing now: schema validation,
artifact registries, feature-manifest checks, shadow execution, and interfaces
that can swap in another conditional joint generator. Do not yet make the
current categorical CVAE a hard dependency for irreversible pipeline choices,
calibration claims, or downstream estimands that rely on its learned residual
dependence.

The next validation cycle should first separate the two observed problems:
compare fixed-checkpoint Gauss-Hermite estimates with scrambled Sobol QMC, then
ablate KL weight, fixed epochs versus fixed optimizer updates, a no-latent
decoder, and checkpoint selection on prior-integrated marginal log score or
Brier score. A hardened workflow must approach the independent-softmax
marginals under the model-aligned DGP, improve with nested sample size, ignore
the latent variable under the rho=0 control, converge numerically, and remain
stable across initializations before this gate changes.

## Required pipeline manifest

Keep a versioned manifest beside every fitted artifact. It must record:

- repository revision, package version, checkpoint/R format version, and an
  artifact checksum;
- the ordered predictor names and their upstream transformations, units,
  numeric types, and missing-value policy;
- the categorical outcome schema in semantic column order, including every
  level label and its zero-based code;
- `x_dim`, `y_dim`, `encoded_y_dim`, latent dimension, runtime versions, and
  inference device;
- training and inference seeds, `n_mc`, `inference_batch_size`, and
  `decoder_batch_size`.

Fail closed when any manifest field differs. Python checkpoints contain fitted
X scaling but do not contain predictor names, so the external feature manifest
is the authority for Python's positional matrix input. Always pass
`feature_names` when producing the R artifact. Categorical R format 2 then
requires incoming X to have the same unique column-name set and reorders those
columns to training order.

Pin the package revision and NumPy/PyTorch versions used to load a Python
checkpoint. The current checkpoint is a versioned inference record, not a
promise that an arbitrary future package will load it unchanged. Load only
trusted checkpoints. The generated `.R` source embeds its decoder and base-R
runtime, but its checksum and the R version should still be recorded.

## Input and output shapes

- `X` is a finite numeric `n x x_dim` matrix. Python treats its columns as
  positional; categorical R format 2 uses embedded names when present.
- Training `Y` is an `n x y_dim` matrix of finite, zero-based integer codes.
  Semantic outcome order and each code's meaning come only from
  `outcome_schema`.
- `predict_params(X)["probabilities"]` is a mapping in schema order. Each
  outcome maps to an `n x K` matrix whose columns follow that outcome's declared
  level order.
- `generate(X, n_samples_per_x=1)` returns an `int32` `n x y_dim` matrix.
  Larger sample counts return `int32` with shape
  `n x n_samples_per_x x y_dim`. R returns the corresponding integer matrix or
  array and names its outcome dimension.
- Two-level categorical outcomes retain this categorical mapping and two-logit
  representation; their distributions, not their parameter arrays or random
  streams, are equivalent to Bernoulli outcomes.

Use `predict_params()` for marginal probabilities and `generate()` for joint
draws. Do not independently sample the marginal probability matrices when the
shared-latent dependence is required.

## R model isolation

An exported R file defines `.multioutcome_cvae_model` and the `cvae_*` helpers
in the environment into which it is sourced. Source each model into its own
environment when a process holds more than one model:

```r
categorical_model <- new.env(parent = baseenv())
sys.source("categorical_cvae.R", envir = categorical_model)

metadata <- categorical_model$cvae_model_metadata()
draws <- categorical_model$cvae_simulate(
  X,
  n_samples_per_x = 100L,
  seed = 123L,
  decoder_batch_size = 10000L
)
```

Do not source multiple generated files into the global environment; later
files overwrite the same model and helper names.

## Randomness, batching, and training restrictions

Only claim exact stochastic repeatability after establishing it on the pinned
runtime and device; a seed alone is not a cross-device determinism guarantee.
Its scope must include the same artifact, seed, and batching arguments.
Changing a batch size can change which random values are assigned to rows even
though it must not change the target distribution. Record batching parameters
rather than treating them as an untracked performance detail. Python inference
currently consumes PyTorch's global RNG, so set and record
`torch.manual_seed()` immediately before a call that must repeat. R callers
should pass `seed` explicitly.

For repeatable model initialization, seed PyTorch before constructing
`CVAETrainer`; `fit(seed=...)` controls training-time draws but occurs after
network construction. Do not use the convenience random-search or TPE wrappers
in a governed external training pipeline until their initialization,
configuration-sampling, and best-validation-loss behavior has dedicated
reproducibility coverage. Use a fixed, reviewed configuration and a strict
train/validation/test split for current experiments.
