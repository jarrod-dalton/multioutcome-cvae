"""
simulate.py

Core simulation utilities and simple diagnostics for the multioutcome_cvae package.

This module provides:
  - Synthetic data generators for different outcome families
  - Simple marginal and correlation summaries
  - A helper to compare real vs generated outcomes

The simulators are intended for:
  - Sanity-checking the CVAE implementation
  - Reproducible examples and tests
  - Toy worlds where the "true" data-generating process is known

They are NOT required to use CVAETrainer on real data; they are convenience
utilities only.
"""

from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utility: random generator helper
# ---------------------------------------------------------------------


def _get_rng(seed: Optional[int]) -> np.random.Generator:
    """
    Internal helper to create a NumPy random generator.

    Parameters
    ----------
    seed : int or None
        If None, uses NumPy's default RNG. Otherwise, creates a generator
        with the given seed for reproducibility.

    Returns
    -------
    rng : numpy.random.Generator
    """
    if seed is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(seed)


# ---------------------------------------------------------------------
# Core simulators
# ---------------------------------------------------------------------


def simulate_bernoulli_data(
    n_samples: int,
    n_features: int,
    n_outcomes: int,
    latent_dim: int = 2,
    logit_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate multivariate Bernoulli outcomes Y with latent dependence and covariates X.

    Data-generating structure (per observation i):

        X_i ~ N(0, I_{n_features})
        Z_i ~ N(0, I_{latent_dim})

        logits_ij = (X_i @ B_x)_j + (Z_i @ B_z)_j + b_j
        p_ij = sigmoid(logit_scale * logits_ij)

        Y_ij ~ Bernoulli(p_ij)

    Parameters
    ----------
    n_samples : int
        Number of observations (rows of X and Y).

    n_features : int
        Number of columns in X (covariates).

    n_outcomes : int
        Number of outcome dimensions (columns of Y).

    latent_dim : int, default=2
        Dimension of the latent variable Z that induces dependence across Y.

    logit_scale : float, default=1.0
        Global multiplier applied to logits before the sigmoid. Larger values
        make the logits more extreme (stronger separation, lower entropy).

    seed : int or None, default=None
        Random seed for reproducibility. If None, uses NumPy's default generator.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Simulated covariate matrix.

    Y : ndarray of shape (n_samples, n_outcomes)
        Simulated binary outcome matrix (0/1).

    params : dict
        Dictionary of true parameters and latent variables, containing:
          - "B_x"  : (n_features, n_outcomes) weight matrix for X
          - "B_z"  : (latent_dim, n_outcomes) weight matrix for Z
          - "b"    : (n_outcomes,) intercept vector
          - "Z"    : (n_samples, latent_dim) latent factors
          - "latent_dim"   : latent dimension
          - "logit_scale"  : scalar used in the simulation
          - "outcome_type" : "bernoulli"
          - "seed"         : seed used
    """
    rng = _get_rng(seed)

    # Covariates and latent factors
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    Z = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)

    # Weights: scale by sqrt to keep logits in a reasonable range
    B_x = rng.normal(scale=1.0 / np.sqrt(n_features), size=(n_features, n_outcomes)).astype(
        np.float32
    )
    B_z = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(latent_dim, n_outcomes)).astype(
        np.float32
    )
    b = rng.normal(scale=0.3, size=(n_outcomes,)).astype(np.float32)

    logits = X @ B_x + Z @ B_z + b  # (n_samples, n_outcomes)
    logits = logit_scale * logits

    probs = 1.0 / (1.0 + np.exp(-logits))
    Y = rng.binomial(n=1, p=probs).astype(np.float32)

    params = {
        "B_x": B_x,
        "B_z": B_z,
        "b": b,
        "Z": Z,
        "latent_dim": latent_dim,
        "logit_scale": logit_scale,
        "outcome_type": "bernoulli",
        "seed": seed,
    }

    return X, Y, params


def simulate_gaussian_data(
    n_samples: int,
    n_features: int,
    n_outcomes: int,
    latent_dim: int = 2,
    noise_sd: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate multivariate Gaussian outcomes Y with latent dependence and covariates X.

    Data-generating structure (per observation i):

        X_i ~ N(0, I_{n_features})
        Z_i ~ N(0, I_{latent_dim})

        mu_ij = (X_i @ B_x)_j + (Z_i @ B_z)_j + b_j
        Y_ij  ~ Normal(mu_ij, noise_sd^2)

    Parameters
    ----------
    n_samples : int
        Number of observations (rows of X and Y).

    n_features : int
        Number of columns in X (covariates).

    n_outcomes : int
        Number of outcome dimensions (columns of Y).

    latent_dim : int, default=2
        Dimension of the latent variable Z that induces dependence across Y.

    noise_sd : float, default=1.0
        Standard deviation of the observational noise added to the mean
        predictions. Larger values imply noisier outcomes.

    seed : int or None, default=None
        Random seed for reproducibility. If None, uses NumPy's default generator.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Simulated covariate matrix.

    Y : ndarray of shape (n_samples, n_outcomes)
        Simulated continuous outcome matrix.

    params : dict
        Dictionary of true parameters and latent variables, containing:
          - "B_x"  : (n_features, n_outcomes) weight matrix for X
          - "B_z"  : (latent_dim, n_outcomes) weight matrix for Z
          - "b"    : (n_outcomes,) intercept vector
          - "Z"    : (n_samples, latent_dim) latent factors
          - "latent_dim"   : latent dimension
          - "noise_sd"     : scalar noise SD used
          - "outcome_type" : "gaussian"
          - "seed"         : seed used
    """
    rng = _get_rng(seed)

    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    Z = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)

    B_x = rng.normal(scale=1.0 / np.sqrt(n_features), size=(n_features, n_outcomes)).astype(
        np.float32
    )
    B_z = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(latent_dim, n_outcomes)).astype(
        np.float32
    )
    b = rng.normal(scale=0.5, size=(n_outcomes,)).astype(np.float32)

    mu = X @ B_x + Z @ B_z + b  # (n_samples, n_outcomes)
    eps = rng.normal(scale=noise_sd, size=mu.shape).astype(np.float32)
    Y = mu + eps

    params = {
        "B_x": B_x,
        "B_z": B_z,
        "b": b,
        "Z": Z,
        "latent_dim": latent_dim,
        "noise_sd": noise_sd,
        "outcome_type": "gaussian",
        "seed": seed,
    }

    return X, Y, params


def simulate_poisson_data(
    n_samples: int,
    n_features: int,
    n_outcomes: int,
    latent_dim: int = 2,
    base_rate: float = 0.5,
    rate_scale: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate multivariate Poisson outcomes Y with latent dependence and covariates X.

    Data-generating structure (per observation i):

        X_i ~ N(0, I_{n_features})
        Z_i ~ N(0, I_{latent_dim})

        log_rate_ij = (X_i @ B_x)_j + (Z_i @ B_z)_j + b_j
        lambda_ij   = exp( log_rate_scale * log_rate_ij ) + base_rate
        Y_ij        ~ Poisson(lambda_ij)

    where base_rate keeps rates away from zero and rate_scale controls overall
    variability in log-rates.

    Parameters
    ----------
    n_samples : int
        Number of observations (rows of X and Y).

    n_features : int
        Number of columns in X (covariates).

    n_outcomes : int
        Number of outcome dimensions (columns of Y).

    latent_dim : int, default=2
        Dimension of the latent variable Z that induces dependence across Y.

    base_rate : float, default=0.5
        Baseline rate added to the exponentiated log-rate to avoid zeros.

    rate_scale : float, default=0.5
        Scale applied to the raw linear predictor before exponentiation to
        keep rates in a reasonable numeric range.

    seed : int or None, default=None
        Random seed for reproducibility. If None, uses NumPy's default generator.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Simulated covariate matrix.

    Y : ndarray of shape (n_samples, n_outcomes)
        Simulated count outcome matrix (non-negative integers).

    params : dict
        Dictionary of true parameters and latent variables, containing:
          - "B_x"        : (n_features, n_outcomes) weight matrix for X
          - "B_z"        : (latent_dim, n_outcomes) weight matrix for Z
          - "b"          : (n_outcomes,) intercept vector
          - "Z"          : (n_samples, latent_dim) latent factors
          - "latent_dim" : latent dimension
          - "base_rate"  : baseline rate used
          - "rate_scale" : scale on the linear predictor
          - "outcome_type": "poisson"
          - "seed"       : seed used
    """
    rng = _get_rng(seed)

    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    Z = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)

    B_x = rng.normal(scale=1.0 / np.sqrt(n_features), size=(n_features, n_outcomes)).astype(
        np.float32
    )
    B_z = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(latent_dim, n_outcomes)).astype(
        np.float32
    )
    b = rng.normal(scale=0.3, size=(n_outcomes,)).astype(np.float32)

    log_rate_raw = X @ B_x + Z @ B_z + b  # (n_samples, n_outcomes)
    log_rate = rate_scale * log_rate_raw

    # Avoid extreme rates by clipping
    log_rate = np.clip(log_rate, -5.0, 5.0)

    lam = np.exp(log_rate) + base_rate
    Y = rng.poisson(lam).astype(np.float32)

    params = {
        "B_x": B_x,
        "B_z": B_z,
        "b": b,
        "Z": Z,
        "latent_dim": latent_dim,
        "base_rate": base_rate,
        "rate_scale": rate_scale,
        "outcome_type": "poisson",
        "seed": seed,
    }

    return X, Y, params


def simulate_cvae_data(
    n_samples: int,
    n_features: int,
    n_outcomes: int,
    latent_dim: int = 2,
    outcome_type: str = "bernoulli",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Wrapper to simulate data for CVAE demonstrations, with a configurable
    outcome family.

    Parameters
    ----------
    n_samples : int
        Number of observations.

    n_features : int
        Number of covariates.

    n_outcomes : int
        Number of outcome dimensions.

    latent_dim : int, default=2
        Latent dimension used to induce dependence across Y.

    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"
        Outcome family to simulate.

    seed : int or None, default=None
        Random seed. Passed to the underlying simulator.

    **kwargs : dict
        Additional keyword arguments forwarded to the underlying simulator:
          - For Bernoulli:
              logit_scale (float)
          - For Gaussian:
              noise_sd (float)
          - For Poisson:
              base_rate (float), rate_scale (float)

    Returns
    -------
    X : ndarray
        Covariate matrix.

    Y : ndarray
        Outcome matrix.

    params : dict
        Dictionary with true parameters (see individual simulators).
    """
    outcome_type = outcome_type.lower()
    if outcome_type == "bernoulli":
        X, Y, params = simulate_bernoulli_data(
            n_samples=n_samples,
            n_features=n_features,
            n_outcomes=n_outcomes,
            latent_dim=latent_dim,
            seed=seed,
            **kwargs,
        )
    elif outcome_type == "gaussian":
        X, Y, params = simulate_gaussian_data(
            n_samples=n_samples,
            n_features=n_features,
            n_outcomes=n_outcomes,
            latent_dim=latent_dim,
            seed=seed,
            **kwargs,
        )
    elif outcome_type == "poisson":
        X, Y, params = simulate_poisson_data(
            n_samples=n_samples,
            n_features=n_features,
            n_outcomes=n_outcomes,
            latent_dim=latent_dim,
            seed=seed,
            **kwargs,
        )
    else:
        raise ValueError(
            "outcome_type must be one of {'bernoulli', 'gaussian', 'poisson'}."
        )

    # Record outcome_type at this level as well
    params = dict(params)  # shallow copy
    params["outcome_type"] = outcome_type
    return X, Y, params


# ---------------------------------------------------------------------
# Diagnostics: marginal summaries and correlation structure
# ---------------------------------------------------------------------


def summarize_binary_matrix(
    Y: np.ndarray,
    name: str = "Y",
    make_plot: bool = False,
) -> Dict[str, Any]:
    """
    Compute simple marginal and correlation summaries for a binary (or numeric) matrix.

    This utility is primarily intended for multivariate Bernoulli Y, but will
    work for any numeric matrix (interpreting values as numeric for the mean
    and correlation).

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_outcomes)
        Outcome matrix (typically 0/1 for Bernoulli).

    name : str, default="Y"
        Label used for titles and returned metadata.

    make_plot : bool, default=False
        If True, creates a Matplotlib figure with:
          - Bar plot of marginal means/prevalences
          - Heatmap of the correlation matrix (diagonal masked)

    Returns
    -------
    summary : dict
        Dictionary containing:
          - "name"        : dataset name
          - "n_samples"   : number of rows
          - "n_outcomes"  : number of columns
          - "means"       : marginal means (prevalences for 0/1)
          - "corr"        : correlation matrix
          - "fig"         : Matplotlib figure or None
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_samples, n_outcomes = Y.shape

    means = Y.mean(axis=0)
    if n_outcomes > 1:
        corr = np.corrcoef(Y, rowvar=False)
    else:
        corr = np.array([[1.0]], dtype=np.float32)

    fig = None
    if make_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar plot of means
        ax0 = axes[0]
        ax0.bar(np.arange(n_outcomes), means)
        ax0.set_title(f"{name}: marginal means")
        ax0.set_xlabel("Outcome index")
        ax0.set_ylabel("Mean")

        # Correlation heatmap with diagonal masked
        ax1 = axes[1]
        corr_masked = corr.copy()
        np.fill_diagonal(corr_masked, np.nan)
        im = ax1.imshow(corr_masked, aspect="auto", origin="lower")
        ax1.set_title(f"{name}: correlation (off-diagonal)")
        ax1.set_xlabel("Outcome index")
        ax1.set_ylabel("Outcome index")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        fig.tight_layout()

    return {
        "name": name,
        "n_samples": n_samples,
        "n_outcomes": n_outcomes,
        "means": means,
        "corr": corr,
        "fig": fig,
    }


def compare_real_vs_generated(
    Y_real: np.ndarray,
    Y_gen: np.ndarray,
    label_real: str = "real",
    label_gen: str = "generated",
    make_plot: bool = False,
) -> Dict[str, Any]:
    """
    Compare marginal and correlation structure between real and generated outcomes.

    Parameters
    ----------
    Y_real : ndarray of shape (n_samples_real, n_outcomes)
        Real (observed) outcomes.

    Y_gen : ndarray of shape (n_samples_gen, n_outcomes)
        Generated outcomes (e.g., from a fitted CVAE).

    label_real : str, default="real"
        Label for the real dataset.

    label_gen : str, default="generated"
        Label for the generated dataset.

    make_plot : bool, default=False
        If True, creates a Matplotlib figure with:
          - Real correlation heatmap (off-diagonal)
          - Generated correlation heatmap (off-diagonal)
          - Difference heatmap (corr_gen - corr_real)

    Returns
    -------
    result : dict
        Dictionary containing:
          - "summary_real" : output from summarize_binary_matrix for real data
          - "summary_gen"  : output from summarize_binary_matrix for generated data
          - "corr_real"    : correlation matrix for real data
          - "corr_gen"     : correlation matrix for generated data
          - "corr_diff"    : corr_gen - corr_real
          - "fig"          : Matplotlib figure or None
    """
    summ_real = summarize_binary_matrix(Y_real, name=label_real, make_plot=False)
    summ_gen = summarize_binary_matrix(Y_gen, name=label_gen, make_plot=False)

    corr_real = summ_real["corr"]
    corr_gen = summ_gen["corr"]

    # Ensure same shape
    if corr_real.shape != corr_gen.shape:
        raise ValueError("Real and generated correlation matrices have different shapes.")

    corr_diff = corr_gen - corr_real

    fig = None
    if make_plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Helper: plot corr with diagonal masked
        def _plot_corr(ax, mat, title: str):
            mat_masked = mat.copy()
            np.fill_diagonal(mat_masked, np.nan)
            im = ax.imshow(mat_masked, aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("Outcome index")
            ax.set_ylabel("Outcome index")
            return im

        im0 = _plot_corr(axes[0], corr_real, f"{label_real} corr")
        im1 = _plot_corr(axes[1], corr_gen, f"{label_gen} corr")
        im2 = _plot_corr(axes[2], corr_diff, "difference (gen - real)")

        # Use a single colorbar for the last plot only, to keep layout simpler
        fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)

        fig.tight_layout()

    return {
        "summary_real": summ_real,
        "summary_gen": summ_gen,
        "corr_real": corr_real,
        "corr_gen": corr_gen,
        "corr_diff": corr_diff,
        "fig": fig,
    }
