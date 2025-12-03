"""
Diagnostics and visualization utilities for multioutcome CVAE models.

This module is intentionally independent of any particular training
framework – most functions operate on plain NumPy arrays of
observations and predictions, or on a CVAETrainer-like object that
exposes `predict_params` / `predict_mean` / `generate`.
"""

from typing import Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "calibration_curve_with_ci",
    "plot_global_calibration",
    "plot_per_outcome_calibration_grid",
    "expected_calibration_error",
    "maximum_calibration_error",
    "dependence_curve",
    "plot_dependence_curve",
    "posterior_predictive_check_gaussian",
    "posterior_predictive_check_poisson",
]


# ---------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------


def calibration_curve_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    outcome_type: str = "bernoulli",
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Compute a calibration curve with simple confidence intervals.

    For Bernoulli outcomes:
        - y_true should be 0/1
        - y_pred should be probabilities in [0, 1]
        - bins are defined over [0, 1]
        - CIs are normal-approximation binomial intervals.

    For Gaussian / Poisson outcomes:
        - y_true, y_pred are real-valued / counts
        - bins are defined over the range of y_pred
        - CIs are normal-approximation intervals for the mean of y_true
          in each bin.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed outcomes.
    y_pred : array-like, shape (n,)
        Predicted *mean* (probability for Bernoulli, mean for Gaussian/Poisson).
    n_bins : int, default=10
        Number of bins used to group predictions.
    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"
        Outcome family, which determines binning and CI formula.
    alpha : float, default=0.05
        Significance level for (1 - alpha) confidence intervals.

    Returns
    -------
    result : dict of np.ndarray
        Keys:
          - "bin_centers" : shape (n_bins,)
          - "mean_pred"   : shape (n_bins,)
          - "mean_true"   : shape (n_bins,)
          - "counts"      : shape (n_bins,)
          - "ci_low"      : shape (n_bins,)
          - "ci_high"     : shape (n_bins,)
        Some entries may be NaN if a bin has no observations.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape

    z = 1.96  # approx 95% normal CI
    n = y_true.shape[0]

    if outcome_type == "bernoulli":
        # Bins on [0, 1]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # Bins over range of predictions
        p_min, p_max = y_pred.min(), y_pred.max()
        if p_min == p_max:
            # Avoid degenerate case: make a tiny range
            p_min -= 0.5
            p_max += 0.5
        bin_edges = np.linspace(p_min, p_max, n_bins + 1)

    bin_ids = np.digitize(y_pred, bin_edges) - 1  # map to [0, n_bins-1]

    mean_pred = np.zeros(n_bins)
    mean_true = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    ci_low = np.full(n_bins, np.nan)
    ci_high = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = (bin_ids == b)
        if not np.any(mask):
            mean_pred[b] = np.nan
            mean_true[b] = np.nan
            continue

        counts[b] = mask.sum()
        mean_pred[b] = y_pred[mask].mean()
        mean_true[b] = y_true[mask].mean()

        if counts[b] <= 1:
            # Too few points for a valid interval
            continue

        if outcome_type == "bernoulli":
            # Binomial proportion CI
            p_hat = mean_true[b]
            se = np.sqrt(p_hat * (1.0 - p_hat) / counts[b])
            ci_low[b] = max(0.0, p_hat - z * se)
            ci_high[b] = min(1.0, p_hat + z * se)
        else:
            # CI for mean of y_true in the bin
            y_bin = y_true[mask]
            se = y_bin.std(ddof=1) / np.sqrt(counts[b])
            ci_low[b] = mean_true[b] - z * se
            ci_high[b] = mean_true[b] + z * se

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "bin_centers": bin_centers,
        "mean_pred": mean_pred,
        "mean_true": mean_true,
        "counts": counts,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def plot_global_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outcome_type: str = "bernoulli",
    n_bins: int = 10,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a global calibration curve with error bars.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)
        Observed outcomes and predicted means.
    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"
    n_bins : int, default=10
    alpha : float, default=0.05
    ax : matplotlib Axes or None
        Axis to draw on; if None, a new figure+axes are created.
    label : str or None
        Label for the curve.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    res = calibration_curve_with_ci(
        y_true=y_true,
        y_pred=y_pred,
        n_bins=n_bins,
        outcome_type=outcome_type,
        alpha=alpha,
    )

    mp = res["mean_pred"]
    mt = res["mean_true"]
    ci_low = res["ci_low"]
    ci_high = res["ci_high"]

    mask = ~np.isnan(mp) & ~np.isnan(mt)

    if label is None:
        label = f"CVAE ({outcome_type})"

    if mask.any():
        yerr = np.vstack(
            [
                mt[mask] - ci_low[mask],
                ci_high[mask] - mt[mask],
            ]
        )
        ax.errorbar(
            mp[mask],
            mt[mask],
            yerr=yerr,
            fmt="o-",
            capsize=3,
            label=label,
        )

    # "Ideal" line y = x is conceptually correct for E[Y|X]
    # if on same scale – for Bernoulli it's the standard reliability line;
    # for Gaussian/Poisson this is E[Y] vs predicted mean, so still meaningful.
    min_val = np.nanmin([mp[mask].min(), mt[mask].min()])
    max_val = np.nanmax([mp[mask].max(), mt[mask].max()])
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="ideal")

    ax.set_xlabel("Mean predicted value (bin)")
    ax.set_ylabel("Empirical mean observed value")
    ax.set_title(f"Global calibration ({outcome_type})")
    ax.legend()
    ax.grid(True)

    return ax


def plot_per_outcome_calibration_grid(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    outcome_type: str = "bernoulli",
    n_bins: int = 10,
    alpha: float = 0.05,
    max_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.0),
) -> plt.Figure:
    """
    Plot a grid of calibration curves, one subplot per outcome dimension.

    Parameters
    ----------
    Y_true, Y_pred : array-like, shape (n, d)
        Observed outcomes and predicted means/probabilities.
    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"
    n_bins : int, default=10
    alpha : float, default=0.05
    max_cols : int, default=3
        Maximum number of columns in the subplot grid.
    figsize_per_panel : (float, float), default=(5, 4)
        Size of each subplot in inches.

    Returns
    -------
    fig : matplotlib Figure
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    assert Y_true.shape == Y_pred.shape

    n_outcomes = Y_true.shape[1]
    n_rows = int(np.ceil(n_outcomes / max_cols))
    n_cols = min(max_cols, n_outcomes)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for j in range(n_outcomes):
        ax = axes[j // n_cols, j % n_cols]
        y_j = Y_true[:, j]
        p_j = Y_pred[:, j]

        res = calibration_curve_with_ci(
            y_true=y_j,
            y_pred=p_j,
            n_bins=n_bins,
            outcome_type=outcome_type,
            alpha=alpha,
        )

        mp = res["mean_pred"]
        mt = res["mean_true"]
        ci_low = res["ci_low"]
        ci_high = res["ci_high"]

        mask = ~np.isnan(mp) & ~np.isnan(mt)
        if mask.any():
            yerr = np.vstack(
                [
                    mt[mask] - ci_low[mask],
                    ci_high[mask] - mt[mask],
                ]
            )
            ax.errorbar(
                mp[mask],
                mt[mask],
                yerr=yerr,
                fmt="o-",
                capsize=3,
            )

        # Diagonal
        if mask.any():
            min_val = np.nanmin([mp[mask].min(), mt[mask].min()])
            max_val = np.nanmax([mp[mask].max(), mt[mask].max()])
        else:
            min_val, max_val = 0.0, 1.0

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="gray",
        )
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"Outcome j = {j}")
        ax.set_xlabel("Mean predicted value (bin)")
        ax.set_ylabel("Empirical mean")

        ax.grid(True)

    # Hide unused panels, if any
    for k in range(n_outcomes, n_rows * n_cols):
        fig.delaxes(axes[k // n_cols, k % n_cols])

    fig.suptitle(f"Per-outcome calibration ({outcome_type})", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ---------------------------------------------------------------------
# Calibration error metrics
# ---------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    outcome_type: str = "bernoulli",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (n_b / n) * |mean_true_b - mean_pred_b|

    where b ranges over bins.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)
    n_bins : int, default=10
    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"

    Returns
    -------
    ece : float
    """
    res = calibration_curve_with_ci(
        y_true=y_true,
        y_pred=y_pred,
        n_bins=n_bins,
        outcome_type=outcome_type,
    )
    mt = res["mean_true"]
    mp = res["mean_pred"]
    counts = res["counts"]

    mask = counts > 0
    if not mask.any():
        return np.nan

    weights = counts[mask] / counts[mask].sum()
    ece = np.sum(weights * np.abs(mt[mask] - mp[mask]))
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    outcome_type: str = "bernoulli",
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max_b |mean_true_b - mean_pred_b|

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)
    n_bins : int, default=10
    outcome_type : {"bernoulli", "gaussian", "poisson"}, default="bernoulli"

    Returns
    -------
    mce : float
    """
    res = calibration_curve_with_ci(
        y_true=y_true,
        y_pred=y_pred,
        n_bins=n_bins,
        outcome_type=outcome_type,
    )
    mt = res["mean_true"]
    mp = res["mean_pred"]
    counts = res["counts"]

    mask = counts > 0
    if not mask.any():
        return np.nan

    mce = np.max(np.abs(mt[mask] - mp[mask]))
    return float(mce)


# ---------------------------------------------------------------------
# SHAP-style dependence curves (1D)
# ---------------------------------------------------------------------


def dependence_curve(
    trainer: Any,
    X: np.ndarray,
    feature_index: int,
    outcome_index: int = 0,
    n_grid: int = 50,
    n_mc: int = 20,
    quantile_range: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate a SHAP/partial-dependence-style curve for one feature
    and one outcome dimension.

    For each grid value of X[:, feature_index], we:
      - Replace that column in X with the grid value,
      - Predict E[Y | X] via trainer.predict_mean,
      - Average the chosen outcome dimension over observations.

    Parameters
    ----------
    trainer : object
        Must implement predict_mean(X, n_mc=...).
    X : np.ndarray, shape (n, p)
        Covariate matrix.
    feature_index : int
        Column index in X to vary.
    outcome_index : int, default=0
        Outcome dimension index j to track.
    n_grid : int, default=50
        Number of grid points over which to vary the feature.
    n_mc : int, default=20
        Number of Monte Carlo samples used inside predict_mean().
    quantile_range : (float, float), default=(0.05, 0.95)
        Use this quantile range of X[:, feature_index] for the grid.

    Returns
    -------
    grid_values : np.ndarray, shape (n_grid,)
    mean_preds : np.ndarray, shape (n_grid,)
        Mean predicted E[Y_j | X] at each grid value.
    """
    X = np.asarray(X, dtype=np.float32)
    x_col = X[:, feature_index]
    q_low, q_high = np.quantile(x_col, quantile_range)
    grid_values = np.linspace(q_low, q_high, n_grid)

    mean_preds = np.zeros(n_grid, dtype=np.float64)
    for i, gv in enumerate(grid_values):
        X_mod = X.copy()
        X_mod[:, feature_index] = gv
        y_mean = trainer.predict_mean(X_mod, n_mc=n_mc)
        mean_preds[i] = float(y_mean[:, outcome_index].mean())

    return grid_values, mean_preds


def plot_dependence_curve(
    trainer: Any,
    X: np.ndarray,
    feature_index: int,
    outcome_index: int = 0,
    n_grid: int = 50,
    n_mc: int = 20,
    quantile_range: Tuple[float, float] = (0.05, 0.95),
    feature_name: Optional[str] = None,
    outcome_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a 1D dependence curve (approximate SHAP-style) for one feature
    and one outcome.

    See `dependence_curve` for details.

    Parameters
    ----------
    trainer, X, feature_index, outcome_index, n_grid, n_mc, quantile_range
        As in dependence_curve.
    feature_name : str or None
        Optional label for the feature axis.
    outcome_name : str or None
        Optional label for the outcome.

    Returns
    -------
    ax : matplotlib Axes
    """
    grid_values, mean_preds = dependence_curve(
        trainer=trainer,
        X=X,
        feature_index=feature_index,
        outcome_index=outcome_index,
        n_grid=n_grid,
        n_mc=n_mc,
        quantile_range=quantile_range,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid_values, mean_preds, "-o")
    if feature_name is None:
        feature_name = f"X[:, {feature_index}]"
    if outcome_name is None:
        outcome_name = f"Y[:, {outcome_index}]"

    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"E[{outcome_name} | X]")
    ax.set_title(f"Dependence of {outcome_name} on {feature_name}")
    ax.grid(True)
    return ax


# ---------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------


def posterior_predictive_check_gaussian(
    trainer: Any,
    X: np.ndarray,
    Y: np.ndarray,
    n_rep: int = 100,
    n_mc_params: int = 20,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Posterior predictive check for Gaussian outcomes.

    Uses the predictive distribution parameters from
    trainer.predict_params(X, n_mc=...) to simulate replicated datasets
    Y_rep ~ p(Y | X, model), and compares summaries (mean, variance)
    to those of the observed Y.

    Parameters
    ----------
    trainer : object
        Must implement predict_params(X, n_mc=...) returning at least
        {"mu": ..., "sigma": ...}.
    X : np.ndarray, shape (n, p)
    Y : np.ndarray, shape (n, d)
    n_rep : int, default=100
        Number of replicated datasets to simulate.
    n_mc_params : int, default=20
        Monte Carlo draws for parameters inside predict_params.
    plot : bool, default=True
        If True, show histograms of replicated summaries with observed
        summary overlaid.

    Returns
    -------
    result : dict
        Contains observed and replicated summary statistics.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    params = trainer.predict_params(X, n_mc=n_mc_params)
    mu = np.asarray(params["mu"], dtype=np.float32)
    sigma = np.asarray(params["sigma"], dtype=np.float32)
    sigma = np.clip(sigma, 1e-6, None)

    n, d = Y.shape
    obs_mean = Y.mean(axis=0)
    obs_var = Y.var(axis=0, ddof=1)

    rep_means = np.zeros((n_rep, d), dtype=np.float32)
    rep_vars = np.zeros((n_rep, d), dtype=np.float32)

    rng = np.random.default_rng(12345)
    for r in range(n_rep):
        eps = rng.standard_normal(size=(n, d)).astype(np.float32)
        Y_rep = mu + sigma * eps
        rep_means[r] = Y_rep.mean(axis=0)
        rep_vars[r] = Y_rep.var(axis=0, ddof=1)

    if plot:
        # Plot distribution of the *average* across dimensions, for simplicity
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(rep_means.mean(axis=1), bins=20, alpha=0.7)
        axes[0].axvline(obs_mean.mean(), color="red", linewidth=2, label="observed")
        axes[0].set_title("PPC (Gaussian): mean(Y)")
        axes[0].set_xlabel("Replicated means (avg over dims)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(rep_vars.mean(axis=1), bins=20, alpha=0.7)
        axes[1].axvline(obs_var.mean(), color="red", linewidth=2, label="observed")
        axes[1].set_title("PPC (Gaussian): var(Y)")
        axes[1].set_xlabel("Replicated variances (avg over dims)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "obs_mean": obs_mean,
        "obs_var": obs_var,
        "rep_means": rep_means,
        "rep_vars": rep_vars,
    }


def posterior_predictive_check_poisson(
    trainer: Any,
    X: np.ndarray,
    Y: np.ndarray,
    n_rep: int = 100,
    n_mc_params: int = 20,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Posterior predictive check for Poisson outcomes.

    Uses predictive rate parameters from trainer.predict_params(X, n_mc=...)
    to simulate replicated datasets Y_rep ~ Poisson(rate), and compares
    summaries (mean, variance) to those of the observed Y.

    Parameters
    ----------
    trainer : object
        Must implement predict_params(X, n_mc=...) returning at least
        {"rate": ...}.
    X : np.ndarray, shape (n, p)
    Y : np.ndarray, shape (n, d)
    n_rep : int, default=100
        Number of replicated datasets to simulate.
    n_mc_params : int, default=20
        Monte Carlo draws inside predict_params.
    plot : bool, default=True
        If True, show histograms of replicated summaries with observed
        summary overlaid.

    Returns
    -------
    result : dict
        Contains observed and replicated summary statistics.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    params = trainer.predict_params(X, n_mc=n_mc_params)
    rate = np.asarray(params["rate"], dtype=np.float32)
    rate = np.clip(rate, 1e-8, None)

    n, d = Y.shape
    obs_mean = Y.mean(axis=0)
    obs_var = Y.var(axis=0, ddof=1)

    rep_means = np.zeros((n_rep, d), dtype=np.float32)
    rep_vars = np.zeros((n_rep, d), dtype=np.float32)

    rng = np.random.default_rng(23456)
    for r in range(n_rep):
        Y_rep = rng.poisson(lam=rate, size=(n, d)).astype(np.float32)
        rep_means[r] = Y_rep.mean(axis=0)
        rep_vars[r] = Y_rep.var(axis=0, ddof=1)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(rep_means.mean(axis=1), bins=20, alpha=0.7)
        axes[0].axvline(obs_mean.mean(), color="red", linewidth=2, label="observed")
        axes[0].set_title("PPC (Poisson): mean(Y)")
        axes[0].set_xlabel("Replicated means (avg over dims)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(rep_vars.mean(axis=1), bins=20, alpha=0.7)
        axes[1].axvline(obs_var.mean(), color="red", linewidth=2, label="observed")
        axes[1].set_title("PPC (Poisson): var(Y)")
        axes[1].set_xlabel("Replicated variances (avg over dims)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "obs_mean": obs_mean,
        "obs_var": obs_var,
        "rep_means": rep_means,
        "rep_vars": rep_vars,
    }
