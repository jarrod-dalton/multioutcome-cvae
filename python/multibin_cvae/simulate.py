
import numpy as np
from typing import Dict, Tuple


def simulate_cvae_data(
    n_samples: int = 20000,
    n_features: int = 5,
    n_outcomes: int = 10,
    latent_dim: int = 2,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Simulate data from a latent-variable model that a CVAE can learn.

    X ~ N(0, I)
    z | X ~ N(mu_z(X), sigma_z^2 * I), mu_z(X) = tanh(X W_z + b_z)
    Y | X, z are Bernoulli with logits = [X, z] W_y + b_y
    """
    rng = np.random.default_rng(seed)

    # 1. Simulate covariates X
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # 2. Latent z | X
    W_z = rng.normal(scale=0.5, size=(n_features, latent_dim))
    b_z = rng.normal(scale=0.5, size=(latent_dim,))
    mu_z = np.tanh(X @ W_z + b_z)  # (n_samples, latent_dim)

    sigma_z = 0.7
    z = mu_z + sigma_z * rng.normal(size=(n_samples, latent_dim))

    # 3. Y | X, z
    H = np.concatenate([X, z], axis=1)
    h_dim = H.shape[1]

    W_y = rng.normal(scale=0.7, size=(h_dim, n_outcomes))
    b_y = rng.normal(scale=0.3, size=(n_outcomes,))

    logits = H @ W_y + b_y
    probs = 1.0 / (1.0 + np.exp(-logits))
    Y = rng.binomial(1, probs)

    params = dict(
        W_z=W_z,
        b_z=b_z,
        sigma_z=sigma_z,
        W_y=W_y,
        b_y=b_y,
    )

    return X, Y, params


def train_val_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 1234,
) -> Dict[str, np.ndarray]:
    """Randomly split (X, Y) into train/val/test sets."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]

    return dict(
        X_train=X[idx_train],
        Y_train=Y[idx_train],
        X_val=X[idx_val],
        Y_val=Y[idx_val],
        X_test=X[idx_test],
        Y_test=Y[idx_test],
    )


def summarize_binary_matrix(Y: np.ndarray, name: str = "Y") -> None:
    """Print marginal probabilities and a small correlation block."""
    n_outcomes = Y.shape[1]

    marginals = Y.mean(axis=0)
    print(f"--- Summary for {name} ---")
    print("Marginal Pr(Y_j = 1):")
    for j in range(n_outcomes):
        print(f"  Outcome {j}: {marginals[j]:.3f}")

    corr = np.corrcoef(Y.T)
    print("\nPairwise correlations (first 5x5 block):")
    print(np.round(corr[:5, :5], 3))


def compare_real_vs_generated(
    Y_real: np.ndarray,
    Y_gen: np.ndarray,
    label_real: str = "Real",
    label_gen: str = "Generated",
) -> None:
    """Compare marginals and correlations between real and generated Y."""
    print("=== Marginal probabilities ===")
    real_marg = Y_real.mean(axis=0)
    gen_marg = Y_gen.mean(axis=0)
    for j in range(Y_real.shape[1]):
        print(
            f"Outcome {j}: {label_real}={real_marg[j]:.3f}, "
            f"{label_gen}={gen_marg[j]:.3f}"
        )

    real_corr = np.corrcoef(Y_real.T)
    gen_corr = np.corrcoef(Y_gen.T)

    print("\n=== Pairwise correlations (first 5x5 block) ===")
    print(f"{label_real} correlations:")
    print(np.round(real_corr[:5, :5], 3))
    print(f"\n{label_gen} correlations:")
    print(np.round(gen_corr[:5, :5], 3))
