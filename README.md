# multibin-cvae

Prototype Conditional Variational Autoencoder (CVAE) for multivariate binary outcomes,
with simulation utilities and a minimal R wrapper (via reticulate).

## Python usage

```python
from multibin_cvae import (
    simulate_cvae_data,
    train_val_test_split,
    summarize_binary_matrix,
    compare_real_vs_generated,
    CVAETrainer,
    tune_cvae_random_search,
)

X, Y, params = simulate_cvae_data()
splits = train_val_test_split(X, Y)
X_train, Y_train = splits["X_train"], splits["Y_train"]
X_val, Y_val     = splits["X_val"],   splits["Y_val"]
X_test, Y_test   = splits["X_test"],  splits["Y_test"]

trainer = CVAETrainer(x_dim=X_train.shape[1], y_dim=Y_train.shape[1])
history = trainer.fit(X_train, Y_train, X_val, Y_val)

Y_gen = trainer.generate(X_test, n_samples_per_x=10, return_probs=False)
```

## R usage (sketch)

See `R/multibinCvae/R/*.R` for example wrappers using `reticulate`.
