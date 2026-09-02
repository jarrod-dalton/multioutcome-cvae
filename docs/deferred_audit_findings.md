# Deferred audit findings

These findings predate or sit outside the mixed-cardinality categorical scope.
They are recorded here so the categorical implementation does not quietly
expand into unrelated behavior changes.

- The MLflow training example refits on the full dataset and then labels a
  subset of those same rows as “held out.” Its reported post-refit metrics are
  therefore in-sample and should not be presented as held-out performance.
- `fit(seed=...)` seeds training-time draws after the trainer has constructed
  the neural network. Reproducible initialization currently requires callers
  to seed PyTorch before constructing `CVAETrainer`.
- The unavailable experimental negative-binomial generation branch appears to
  pass NumPy-style success probabilities to PyTorch's differently
  parameterized `NegativeBinomial` sampler. Confirm and correct that branch
  before making the family public.
- Deprecated Poisson fitting applies the common finite-matrix check but does
  not yet enforce non-negative integer outcomes at the family boundary.
- The reticulate-based R examples and package wrappers remain useful legacy
  material, but now overlap with the dependency-free exported-R inference
  path and should be reviewed or clearly separated in a later cleanup.

None of these items is changed by the categorical release candidate.
