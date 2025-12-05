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
    "posterior_predictive_check_neg_binomial",
    "conditional_ppc_by_feature_decile",
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
    trainer,
    X: np.ndarray,
    Y: np.ndarray,
    n_rep: int = 100,
    n_mc_params: int = 20,
    plot: bool = True,
):
    """
    Posterior predictive check for Gaussian CVAE.

    We compare global mean and variance of Y with those from replicated
    datasets drawn from the fitted CVAE.

    Returns a dict with:
      - mean_rep : array of replicated means (length n_rep)
      - var_rep  : array of replicated variances (length n_rep)
      - mean_obs : observed global mean of Y
      - var_obs  : observed global variance of Y
      - fig      : matplotlib Figure or None
    """
    assert trainer.outcome_type == "gaussian", \
        "posterior_predictive_check_gaussian requires outcome_type='gaussian'"

    Y = np.asarray(Y, dtype=np.float32)
    mean_obs = float(Y.mean())
    var_obs = float(Y.var(ddof=0))

    rep_means = []
    rep_vars = []

    for _ in range(n_rep):
        Y_rep = trainer.generate(X, n_samples_per_x=1, return_probs=False)
        rep_means.append(float(Y_rep.mean()))
        rep_vars.append(float(Y_rep.var(ddof=0)))

    rep_means = np.asarray(rep_means)
    rep_vars = np.asarray(rep_vars)

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(rep_means, bins=30)
        axes[0].axvline(mean_obs, color="red", lw=2, label="observed")
        axes[0].set_title("PPC (Gaussian): mean(Y)")
        axes[0].set_xlabel("Replicated means (avg over dims)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(rep_vars, bins=30)
        axes[1].axvline(var_obs, color="red", lw=2, label="observed")
        axes[1].set_title("PPC (Gaussian): var(Y)")
        axes[1].set_xlabel("Replicated variances (avg over dims)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        fig.tight_layout()

    return {
        "mean_rep": rep_means,
        "var_rep": rep_vars,
        "mean_obs": mean_obs,
        "var_obs": var_obs,
        "fig": fig,
    }

def posterior_predictive_check_poisson(
    trainer,
    X: np.ndarray,
    Y: np.ndarray,
    n_rep: int = 100,
    n_mc_params: int = 20,
    plot: bool = True,
):
    """
    Posterior predictive check for Poisson CVAE.

    Compares global mean and variance of Y with those from replicated
    datasets drawn from the fitted Poisson CVAE.

    Returns the same keys as the Gaussian PPC:
      - mean_rep, var_rep, mean_obs, var_obs, fig
    """
    assert trainer.outcome_type == "poisson", \
        "posterior_predictive_check_poisson requires outcome_type='poisson'"

    Y = np.asarray(Y, dtype=np.float32)
    mean_obs = float(Y.mean())
    var_obs = float(Y.var(ddof=0))

    rep_means = []
    rep_vars = []

    for _ in range(n_rep):
        Y_rep = trainer.generate(X, n_samples_per_x=1, return_probs=False)
        rep_means.append(float(Y_rep.mean()))
        rep_vars.append(float(Y_rep.var(ddof=0)))

    rep_means = np.asarray(rep_means)
    rep_vars = np.asarray(rep_vars)

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(rep_means, bins=30)
        axes[0].axvline(mean_obs, color="red", lw=2, label="observed")
        axes[0].set_title("PPC (Poisson): mean(Y)")
        axes[0].set_xlabel("Replicated means (avg over dims)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(rep_vars, bins=30)
        axes[1].axvline(var_obs, color="red", lw=2, label="observed")
        axes[1].set_title("PPC (Poisson): var(Y)")
        axes[1].set_xlabel("Replicated variances (avg over dims)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        fig.tight_layout()

    return {
        "mean_rep": rep_means,
        "var_rep": rep_vars,
        "mean_obs": mean_obs,
        "var_obs": var_obs,
        "fig": fig,
    }

def posterior_predictive_check_neg_binomial(
    trainer,
    X: np.ndarray,
    Y: np.ndarray,
    n_rep: int = 100,
    n_mc_params: int = 20,
    plot: bool = True,
):
    """
    Posterior predictive check for Negative Binomial CVAE.

    Same interface as the Gaussian/Poisson PPCs.
    """
    assert trainer.outcome_type == "neg_binomial", \
        "posterior_predictive_check_neg_binomial requires outcome_type='neg_binomial'"

    Y = np.asarray(Y, dtype=np.float32)
    mean_obs = float(Y.mean())
    var_obs = float(Y.var(ddof=0))

    rep_means = []
    rep_vars = []

    for _ in range(n_rep):
        Y_rep = trainer.generate(X, n_samples_per_x=1, return_probs=False)
        rep_means.append(float(Y_rep.mean()))
        rep_vars.append(float(Y_rep.var(ddof=0)))

    rep_means = np.asarray(rep_means)
    rep_vars = np.asarray(rep_vars)

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(rep_means, bins=30)
        axes[0].axvline(mean_obs, color="red", lw=2, label="observed")
        axes[0].set_title("PPC (NegBin): mean(Y)")
        axes[0].set_xlabel("Replicated means (avg over dims)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(rep_vars, bins=30)
        axes[1].axvline(var_obs, color="red", lw=2, label="observed")
        axes[1].set_title("PPC (NegBin): var(Y)")
        axes[1].set_xlabel("Replicated variances (avg over dims)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        fig.tight_layout()

    return {
        "mean_rep": rep_means,
        "var_rep": rep_vars,
        "mean_obs": mean_obs,
        "var_obs": var_obs,
        "fig": fig,
    }

def conditional_ppc_by_feature_decile(
    trainer,
    X,
    Y,
    feature_index: int = 1,
    n_bins: int = 10,
    n_rep: int = 100,
    random_state: int = 123,
    plot: bool = True,
):
    """
    SHAP-style posterior predictive check by deciles of a single X feature.

    For each decile of X[:, feature_index], we:
      - compute the observed global mean and variance of Y (averaged over all
        outcomes and rows in that decile);
      - draw replicated datasets from the fitted CVAE and compute the same
        summaries, building up a posterior predictive distribution.

    Parameters
    ----------
    trainer : CVAETrainer
        Fitted trainer object (typically outcome_type='neg_binomial' or 'poisson').

    X : array-like, shape (n_samples, x_dim)
        Covariate matrix used for fitting (or a held-out set).

    Y : array-like, shape (n_samples, y_dim)
        Outcome matrix corresponding to X.

    feature_index : int, default=1
        Column index of X to slice on (0-based). For "X[2]" use feature_index=2.

    n_bins : int, default=10
        Number of quantile bins (e.g., deciles).

    n_rep : int, default=100
        Number of replicated datasets to draw for each bin.

    random_state : int, default=123
        Seed for reproducibility of the PPC draws.

    plot : bool, default=True
        If True, produces a figure with:
          - dependence plot for the mean
          - dependence plot for the variance

    Returns
    -------
    result : dict
        {
          "bin_edges": 1D array of bin edges (length n_bins + 1),
          "bin_centers": 1D array of bin centers (length n_bins),
          "mean_obs": 1D array of observed global means per bin,
          "var_obs": 1D array of observed global variances per bin,
          "mean_rep": 2D array (n_rep, n_bins) of replicated means,
          "var_rep": 2D array (n_rep, n_bins) of replicated variances,
          "fig": matplotlib Figure or None
        }
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    rng = np.random.default_rng(random_state)

    x_col = X[:, feature_index]
    # Quantile bin edges
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(x_col, qs)
    # Ensure strictly increasing edges (in case of ties)
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    bin_idx = np.digitize(x_col, bin_edges) - 1  # 0..n_bins-1

    mean_obs = np.full(n_bins, np.nan)
    var_obs = np.full(n_bins, np.nan)
    mean_rep = np.full((n_rep, n_bins), np.nan)
    var_rep  = np.full((n_rep, n_bins), np.nan)

    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        X_b = X[mask]
        Y_b = Y[mask]

        # observed summaries (global over all outcomes in the slice)
        mean_obs[b] = Y_b.mean()
        var_obs[b]  = Y_b.var()

        # replicated summaries
        for r in range(n_rep):
            # 1 draw per row; trainer.generate already samples over z
            Y_rep_b = trainer.generate(X_b, n_samples_per_x=1, return_probs=False)
            mean_rep[r, b] = Y_rep_b.mean()
            var_rep[r, b]  = Y_rep_b.var()

    # Bin centers for plotting on x-axis
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        ax_mean, ax_var = axes

        def _plot_dep(ax, rep_mat, obs_vec, ylabel: str, title: str):
            # rep_mat: (n_rep, n_bins)
            q_low, q_med, q_hi = np.nanpercentile(rep_mat, [5, 50, 95], axis=0)

            ax.fill_between(
                bin_centers,
                q_low,
                q_hi,
                alpha=0.3,
                linewidth=0,
                label="PPC 5–95%",
            )
            ax.plot(
                bin_centers,
                q_med,
                linestyle="-",
                marker="o",
                label="PPC median",
            )
            ax.plot(
                bin_centers,
                obs_vec,
                linestyle="--",
                marker="x",
                color="black",
                label="observed",
            )
            ax.set_xlabel(f"X[:, {feature_index}] (bin center)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        _plot_dep(
            ax_mean,
            mean_rep,
            mean_obs,
            ylabel="Global mean of Y",
            title="Conditional PPC: mean(Y) by X feature decile",
        )

        _plot_dep(
            ax_var,
            var_rep,
            var_obs,
            ylabel="Global variance of Y",
            title="Conditional PPC: var(Y) by X feature decile",
        )

        handles, labels = ax_mean.get_legend_handles_labels()
        ax_mean.legend(handles, labels, loc="best")
        fig.tight_layout()

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "mean_obs": mean_obs,
        "var_obs": var_obs,
        "mean_rep": mean_rep,
        "var_rep": var_rep,
        "fig": fig,
    }
