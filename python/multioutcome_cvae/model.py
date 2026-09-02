import numpy as np
from typing import Optional, Dict, Any, List
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.special import gammaln  # PyTorch >= 1.8

# ---------------------------------------------------------------------------
# Production-supported outcomes are "bernoulli" (binary Y), "categorical"
# (mixed-cardinality nominal Y), and "gaussian" (normal Y). "poisson"
# (count Y) remains available for compatibility but is deprecated.
#
# In experimentation, the CVAE performed reasonably well for generated 
# binary and continuous data. The Poisson model estimated conditional
# means pretty well, but tended to under-estimate global variance
# (overdispersion). Negative binomial is theoretically a good option in that
# situation and is implemented in this file, but in experimentation
# there was a lot of difficulty getting the CVAE to converge to
# the right mean and variance parameters so it is not made available as
# a valid outcome type currently.
# ---------------------------------------------------------------------------

# Production-supported outcome families
PRODUCTION_OUTCOME_TYPES = ("bernoulli", "gaussian", "categorical")

# Retained for compatibility, without a production guarantee
DEPRECATED_OUTCOME_TYPES = ("poisson",)

VALID_OUTCOME_TYPES = PRODUCTION_OUTCOME_TYPES + DEPRECATED_OUTCOME_TYPES

# Experimental / unstable:
EXPERIMENTAL_OUTCOME_TYPES = ("neg_binomial",)


def _normalize_outcome_schema(
    outcome_type: str,
    y_dim: int,
    outcome_schema: Optional[List[Dict[str, Any]]],
):
    """Validate categorical metadata and derive its flat decoder layout."""
    if outcome_type != "categorical":
        if outcome_schema is not None:
            raise ValueError(
                "outcome_schema is only valid for outcome_type='categorical'."
            )
        return None, y_dim, []

    if outcome_schema is None:
        raise ValueError(
            "outcome_schema is required for outcome_type='categorical'."
        )
    if not isinstance(outcome_schema, (list, tuple)):
        raise ValueError("outcome_schema must be a list of outcome dictionaries.")
    if len(outcome_schema) != y_dim:
        raise ValueError(
            "outcome_schema must contain exactly one entry per semantic "
            f"outcome ({y_dim}); received {len(outcome_schema)}."
        )

    normalized = []
    names = set()
    slices = []
    start = 0
    for index, entry in enumerate(outcome_schema):
        if not isinstance(entry, dict):
            raise ValueError(
                f"outcome_schema[{index}] must be a dictionary with name and levels."
            )

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"outcome_schema[{index}].name must be a non-empty string."
            )
        if name in names:
            raise ValueError(f"Duplicate outcome name in outcome_schema: {name!r}.")
        names.add(name)

        levels = entry.get("levels")
        if not isinstance(levels, (list, tuple)):
            raise ValueError(
                f"outcome_schema[{index}].levels must be a list of strings."
            )
        if len(levels) < 2:
            raise ValueError(
                f"Outcome {name!r} must define at least two levels."
            )
        normalized_levels = []
        seen_levels = set()
        for level_index, level in enumerate(levels):
            if not isinstance(level, str) or not level.strip():
                raise ValueError(
                    f"outcome_schema[{index}].levels[{level_index}] must be "
                    "a non-empty string."
                )
            if level in seen_levels:
                raise ValueError(
                    f"Outcome {name!r} contains duplicate level {level!r}."
                )
            seen_levels.add(level)
            normalized_levels.append(level)

        stop = start + len(normalized_levels)
        normalized.append({"name": name, "levels": normalized_levels})
        slices.append((start, stop))
        start = stop

    return normalized, start, slices

# ---------------------------------------------------------------------------
# Log-likelihood helper functions (used in tests and potentially by users)
# ---------------------------------------------------------------------------
def _bernoulli_loglik(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Sum log-likelihood for Bernoulli outcomes with given logits.

    Parameters
    ----------
    y : torch.Tensor
        Observed 0/1 outcomes, same shape as logits.
    logits : torch.Tensor
        Logits (real-valued) for Bernoulli probabilities.

    Returns
    -------
    torch.Tensor (scalar)
        Sum over all entries of log p(y | logits).
    """
    # BCE with logits is -loglik; use elementwise and negate
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loglik = -bce
    return loglik.sum()


def _gaussian_loglik(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Sum log-likelihood for Gaussian outcomes Y ~ N(mu, sigma^2).

    Parameters
    ----------
    y : torch.Tensor
        Observed values.
    mu : torch.Tensor
        Mean parameter, same shape as y.
    sigma : torch.Tensor
        Standard deviation parameter (> 0), same shape as y.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    torch.Tensor (scalar)
        Sum over all entries of log p(y | mu, sigma).
    """
    sigma = torch.clamp(sigma, min=eps)
    var = sigma ** 2
    log_sigma = torch.log(sigma)
    # log N(y; mu, sigma^2) = -0.5 * [ (y-mu)^2/var + 2 log sigma + log(2π) ]
    ll_mat = -0.5 * (((y - mu) ** 2) / var + 2.0 * log_sigma + np.log(2.0 * np.pi))
    return ll_mat.sum()


def _poisson_loglik(
    y: torch.Tensor,
    rate: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Sum log-likelihood for Poisson outcomes Y ~ Poisson(rate).

    Parameters
    ----------
    y : torch.Tensor
        Observed counts (>= 0).
    rate : torch.Tensor
        Poisson rate (lambda), same shape as y.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    torch.Tensor (scalar)
        Sum over all entries of log p(y | rate).
    """
    rate = torch.clamp(rate, min=eps)
    ll_mat = y * torch.log(rate) - rate - torch.lgamma(y + 1.0)
    return ll_mat.sum()


# ---------------------------------------------------------------------------
# Negative binomial NLL with softplus parameterization
# ---------------------------------------------------------------------------


def _neg_binomial_nll(
    y: torch.Tensor,
    raw_mu: torch.Tensor,
    raw_r: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative binomial negative log-likelihood (sum over all elements).

    We parameterize via unconstrained raw_mu, raw_r and map to
        mu = softplus(raw_mu) > 0
        r  = softplus(raw_r)  > 0

    So that
        Var(Y | X) = mu + mu^2 / r
    """

    # Stabilized positive parameters
    mu = F.softplus(raw_mu) + eps
    r = F.softplus(raw_r) + eps

    # log p(Y = y) under NB(r, p) with mean mu:
    # p = r / (r + mu)
    # log p(y) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
    #            + r * log(r / (r + mu)) + y * log(mu / (r + mu))
    t1 = gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)

    log_r_over_rplusmu = torch.log(r) - torch.log(r + mu)
    log_mu_over_rplusmu = torch.log(mu) - torch.log(r + mu)

    t2 = r * log_r_over_rplusmu
    t3 = y * log_mu_over_rplusmu

    log_p = t1 + t2 + t3

    if mask is not None:
        log_p = log_p * mask

    # Negative log-likelihood (sum over all entries)
    return -torch.sum(log_p)



class XYDataset(Dataset):
    """
    Basic dataset wrapper for (X, Y) with optional mask over Y.

    Parameters
    ----------
    X : np.ndarray, shape (n, x_dim)
    Y : np.ndarray, shape (n, y_dim)
    mask : np.ndarray or None, shape (n, y_dim)
        Optional mask over Y: 1 = observed, 0 = missing.
        If None, all entries are treated as observed.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, mask: Optional[np.ndarray] = None):
        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[0] == Y.shape[0]

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        if mask is not None:
            mask = np.asarray(mask)
            assert mask.shape == Y.shape, "Y mask must have same shape as Y."
            # store as float32 so we can multiply with loss terms
            self.mask = mask.astype(np.float32)
        else:
            self.mask = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.mask is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx], self.Y[idx], self.mask[idx]

class MultivariateOutcomeCVAE(nn.Module):
    """
    Conditional VAE for multivariate outcomes with selectable family:

      outcome_type ∈ {"bernoulli", "gaussian", "categorical", "poisson",
      "neg_binomial"}

    Encoder: q(z | x, y)
    Decoder: p(y | x, z)
    Prior:   p(z) = N(0, I)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int = 8,
        outcome_type: str = "bernoulli",
        enc_hidden_dims: Optional[List[int]] = None,
        dec_hidden_dims: Optional[List[int]] = None,
        outcome_schema: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        assert outcome_type in VALID_OUTCOME_TYPES, \
            f"outcome_type must be one of {VALID_OUTCOME_TYPES}"
        if outcome_type in DEPRECATED_OUTCOME_TYPES:
            warnings.warn(
                "Poisson CVAE support is deprecated and is not production-supported.",
                FutureWarning,
                stacklevel=2,
            )
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.outcome_type = outcome_type
        (
            self.outcome_schema,
            self.encoded_y_dim,
            self.outcome_slices,
        ) = _normalize_outcome_schema(outcome_type, y_dim, outcome_schema)

        if enc_hidden_dims is None or len(enc_hidden_dims) == 0:
            enc_hidden_dims = [64, 64]
        if dec_hidden_dims is None or len(dec_hidden_dims) == 0:
            dec_hidden_dims = enc_hidden_dims

        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims

        # ---------------- Encoder over [x, y] ----------------
        enc_input_dim = x_dim + self.encoded_y_dim
        self.enc_layers = nn.ModuleList()
        in_dim = enc_input_dim
        for h_dim in enc_hidden_dims:
            self.enc_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        enc_last_dim = in_dim

        self.enc_mu = nn.Linear(enc_last_dim, latent_dim)
        self.enc_logvar = nn.Linear(enc_last_dim, latent_dim)

        # ---------------- Decoder core over [x, z] ----------------
        dec_input_dim = x_dim + latent_dim
        self.dec_layers = nn.ModuleList()
        in_dim = dec_input_dim
        for h_dim in dec_hidden_dims:
            self.dec_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        dec_last_dim = in_dim

        # A small hidden size for the per-family heads
        head_hidden = max(32, dec_last_dim // 2)

        # ---------------- Family-specific heads + X skip ----------------
        if outcome_type == "bernoulli":
            # logits(x, z) = head(h) + W_skip x
            self.dec_logits_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, y_dim),
            )
            self.dec_logits_skip = nn.Linear(x_dim, y_dim)

        elif outcome_type == "categorical":
            # One contiguous logits block per semantic outcome.
            self.dec_logits_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, self.encoded_y_dim),
            )
            self.dec_logits_skip = nn.Linear(x_dim, self.encoded_y_dim)

        elif outcome_type == "gaussian":
            # mu(x, z) = head_mu(h) + W_skip_mu x
            # logvar(x, z) = head_logvar(h)
            self.dec_mu_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, y_dim),
            )
            self.dec_mu_skip = nn.Linear(x_dim, y_dim)

            self.dec_logvar_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, y_dim),
            )

        elif outcome_type == "poisson":
            # log_rate(x, z) = head(h) + W_skip x
            self.dec_log_rate_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, y_dim),
            )
            self.dec_log_rate_skip = nn.Linear(x_dim, y_dim)

        elif outcome_type == "neg_binomial":
            # raw_mu(x, z) = head(h) + W_skip x   (later softplus → mu > 0)
            self.dec_raw_mu_head = nn.Sequential(
                nn.Linear(dec_last_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, y_dim),
            )
            self.dec_raw_mu_skip = nn.Linear(x_dim, y_dim)

            # Global per-outcome dispersion r_j (unconstrained; later softplus)
            self.dec_raw_r_global = nn.Parameter(torch.zeros(y_dim))

        else:
            raise ValueError("Invalid outcome_type.")

    # ---------------- Encode / reparameterize / decode ----------------

    def _encode_categorical_y(
        self,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(
                f"Categorical y must have shape (n, {self.y_dim})."
            )
        if not torch.isfinite(y).all():
            raise ValueError("Categorical y must contain only finite values.")
        if not torch.equal(y, torch.round(y)):
            raise ValueError("Categorical y codes must be integers.")
        if mask is not None:
            if mask.shape != y.shape:
                raise ValueError("Categorical mask must have the same shape as y.")
            if not torch.isfinite(mask).all() or not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise ValueError("Categorical mask values must be exactly 0 or 1.")

        groups = []
        for outcome_index, schema_entry in enumerate(self.outcome_schema):
            cardinality = len(schema_entry["levels"])
            codes = y[:, outcome_index].long()
            if torch.any(codes < 0) or torch.any(codes >= cardinality):
                raise ValueError(
                    f"Categorical y codes for {schema_entry['name']!r} must be "
                    f"between 0 and {cardinality - 1}."
                )
            group = F.one_hot(codes, num_classes=cardinality).to(dtype=y.dtype)
            if mask is not None:
                group = group * mask[:, outcome_index].unsqueeze(1).to(y.dtype)
            groups.append(group)
        return torch.cat(groups, dim=1)

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.outcome_type == "categorical":
            y_for_encoder = self._encode_categorical_y(y, mask=mask).to(x.dtype)
        else:
            y_for_encoder = y
        h = torch.cat([x, y_for_encoder], dim=1)
        for layer in self.enc_layers:
            h = F.relu(layer(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decoder p(y | x, z). Uses a shared core over [x, z] followed by
        family-specific heads with X skip connections.
        """
        h = torch.cat([x, z], dim=1)
        for layer in self.dec_layers:
            h = F.relu(layer(h))

        if self.outcome_type in ("bernoulli", "categorical"):
            logits_core = self.dec_logits_head(h)
            logits_skip = self.dec_logits_skip(x)
            logits = logits_core + logits_skip
            return {"logits": logits}

        elif self.outcome_type == "gaussian":
            mu_core = self.dec_mu_head(h)
            mu_skip = self.dec_mu_skip(x)
            mu = mu_core + mu_skip

            logvar = self.dec_logvar_head(h)
            return {"mu": mu, "logvar": logvar}

        elif self.outcome_type == "poisson":
            log_rate_core = self.dec_log_rate_head(h)
            log_rate_skip = self.dec_log_rate_skip(x)
            log_rate = log_rate_core + log_rate_skip
            return {"log_rate": log_rate}

        elif self.outcome_type == "neg_binomial":
            # Per-sample raw_mu with X skip
            raw_mu_core = self.dec_raw_mu_head(h)
            raw_mu_skip = self.dec_raw_mu_skip(x)
            raw_mu = raw_mu_core + raw_mu_skip

            # Global per-outcome raw_r, broadcast across batch
            batch_size = x.size(0)
            raw_r = self.dec_raw_r_global.unsqueeze(0).expand(batch_size, -1)

            return {"raw_mu": raw_mu, "raw_r": raw_r}

        else:
            raise ValueError("Invalid outcome_type.")

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Encode -> reparameterize -> decode.
        Returns:
            out: decoder outputs (family-specific)
            mu_z, logvar_z: parameters of q(z | x, y)
        """
        mu_z, logvar_z = self.encode(x, y, mask=mask)
        z = self.reparameterize(mu_z, logvar_z)
        out = self.decode(x, z)
        return out, mu_z, logvar_z


class CVAETrainer:
    """
    CVAETrainer

    High-level wrapper around MultivariateOutcomeCVAE. Handles:

    - construction of encoder/decoder neural networks
    - standardization of X
    - training loop (reconstruction + KL)
    - prediction and generation helpers
    - outcome-family-specific behavior
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int = 8,
        outcome_type: str = "bernoulli",
        enc_hidden_dims: Optional[List[int]] = None,
        dec_hidden_dims: Optional[List[int]] = None,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        device: Optional[str] = None,
        outcome_schema: Optional[List[Dict[str, Any]]] = None,
    ):
        if outcome_type not in VALID_OUTCOME_TYPES + EXPERIMENTAL_OUTCOME_TYPES:
            raise ValueError(
                f"outcome_type must be one of {VALID_OUTCOME_TYPES + EXPERIMENTAL_OUTCOME_TYPES}"
            )
        self.outcome_type = outcome_type
        self.experimental_nb = (outcome_type in EXPERIMENTAL_OUTCOME_TYPES)

        if self.experimental_nb:
            raise NotImplementedError(
                "Outcome type 'neg_binomial' is experimental and not yet "
                "supported in the public API. This release only supports "
                "outcome_type in {'bernoulli', 'categorical', 'gaussian', "
                "'poisson'}."
            )
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.outcome_type = outcome_type
        (
            self.outcome_schema,
            self.encoded_y_dim,
            self.outcome_slices,
        ) = _normalize_outcome_schema(outcome_type, y_dim, outcome_schema)

        if enc_hidden_dims is None:
            enc_hidden_dims = [hidden_dim] * n_hidden_layers
        if dec_hidden_dims is None:
            dec_hidden_dims = enc_hidden_dims

        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta_kl = beta_kl

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = MultivariateOutcomeCVAE(
            x_dim=x_dim,
            y_dim=y_dim,
            latent_dim=latent_dim,
            outcome_type=outcome_type,
            enc_hidden_dims=enc_hidden_dims,
            dec_hidden_dims=dec_hidden_dims,
            outcome_schema=self.outcome_schema,
        ).to(self.device)

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.trained: bool = False

    # --------- standardization helpers ---------
    @staticmethod
    def _validate_matrix(
        value: np.ndarray,
        name: str,
        expected_cols: int,
    ) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric matrix.") from exc

        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix.")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row.")
        if array.shape[1] != expected_cols:
            raise ValueError(
                f"{name} must contain exactly {expected_cols} columns; "
                f"received {array.shape[1]}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def _validate_outcomes(self, Y: np.ndarray, name: str) -> np.ndarray:
        if self.outcome_type == "categorical":
            # Validate in float64 before the training representation is narrowed
            # to float32. Otherwise, sufficiently small fractional parts could
            # round away and violate the integer-code contract unnoticed.
            try:
                categorical_y = np.asarray(Y, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a numeric matrix.") from exc
            if categorical_y.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional matrix.")
            if categorical_y.shape[0] == 0:
                raise ValueError(f"{name} must contain at least one row.")
            if categorical_y.shape[1] != self.y_dim:
                raise ValueError(
                    f"{name} must contain exactly {self.y_dim} columns; "
                    f"received {categorical_y.shape[1]}."
                )
            if not np.isfinite(categorical_y).all():
                raise ValueError(f"{name} must contain only finite values.")
            if not np.equal(categorical_y, np.floor(categorical_y)).all():
                raise ValueError("Categorical outcome codes must be integers.")
            for outcome_index, schema_entry in enumerate(self.outcome_schema):
                cardinality = len(schema_entry["levels"])
                codes = categorical_y[:, outcome_index]
                if np.any(codes < 0) or np.any(codes >= cardinality):
                    raise ValueError(
                        f"Categorical outcome codes for {schema_entry['name']!r} "
                        f"must be between 0 and {cardinality - 1}."
                    )
            return categorical_y.astype(np.float32)

        Y = self._validate_matrix(Y, name, self.y_dim)
        if self.outcome_type == "bernoulli" and not np.isin(Y, (0.0, 1.0)).all():
            raise ValueError("Bernoulli outcomes must be exactly 0 or 1.")
        return Y

    def _validate_categorical_mask(
        self,
        mask: Optional[np.ndarray],
        name: str,
        expected_rows: int,
    ) -> Optional[np.ndarray]:
        if mask is None or self.outcome_type != "categorical":
            return mask
        try:
            array = np.asarray(mask, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric matrix.") from exc
        if array.ndim != 2 or array.shape != (expected_rows, self.y_dim):
            raise ValueError(
                f"{name} must have shape ({expected_rows}, {self.y_dim}); "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
            raise ValueError(f"{name} values must be finite and exactly 0 or 1.")
        return array

    def _fit_standardizer(self, X_train: np.ndarray):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        self.x_mean = mean.astype(np.float32)
        self.x_std = std.astype(np.float32)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize X using parameters fitted from the training data.
        """
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError(
                "The X standardizer has not been fitted. Call fit() first."
            )
        X = self._validate_matrix(X, "X", self.x_dim)
        return (X - self.x_mean) / self.x_std

    # --------- reconstruction loss by outcome family ---------
    def _recon_loss(
        self,
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.outcome_type == "bernoulli":
            logits = out["logits"]
            if mask is not None:
                return F.binary_cross_entropy_with_logits(
                    logits, y, weight=mask, reduction="sum"
                )
            else:
                return F.binary_cross_entropy_with_logits(
                    logits, y, reduction="sum"
                )

        elif self.outcome_type == "gaussian":
            mu = out["mu"]
            logvar = out["logvar"]
            if mask is not None:
                return 0.5 * torch.sum(
                    mask * (logvar + (y - mu) ** 2 / torch.exp(logvar))
                )
            else:
                return 0.5 * torch.sum(
                    logvar + (y - mu) ** 2 / torch.exp(logvar)
                )

        elif self.outcome_type == "poisson":
            log_rate = out["log_rate"]
            rate = torch.exp(log_rate)
            if mask is not None:
                return torch.sum(mask * (rate - y * log_rate))
            else:
                return torch.sum(rate - y * log_rate)

        elif self.outcome_type == "categorical":
            logits = out["logits"]
            loss = logits.new_zeros(())
            for outcome_index, (start, stop) in enumerate(self.outcome_slices):
                outcome_loss = F.cross_entropy(
                    logits[:, start:stop],
                    y[:, outcome_index].long(),
                    reduction="none",
                )
                if mask is not None:
                    outcome_loss = outcome_loss * mask[:, outcome_index]
                loss = loss + outcome_loss.sum()
            return loss

        elif self.outcome_type == "neg_binomial":
            raw_mu = out["raw_mu"]
            raw_r = out["raw_r"]
            return _neg_binomial_nll(y, raw_mu, raw_r, mask=mask)

        else:
            raise ValueError("Invalid outcome_type for recon loss.")

    # --------- training ---------
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        beta_kl: Optional[float] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        Y_mask_train: Optional[np.ndarray] = None,
        Y_mask_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        kl_warmup_epochs: int = 0,
        max_grad_norm: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Fit the CVAE.

        New arguments
        -------------
        Y_mask_train : np.ndarray or None, shape (n_train, y_dim)
            Optional mask over Y_train: 1 = observed, 0 = missing. If None,
            all entries are treated as observed.

        Y_mask_val : np.ndarray or None, shape (n_val, y_dim)
            Optional mask over Y_val, used for validation loss.
        """
        # Allow both num_epochs and epochs; epochs is a simple alias used in tests.
        if epochs is not None:
            num_epochs = epochs

        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        lr = lr if lr is not None else self.lr
        beta_kl = beta_kl if beta_kl is not None else self.beta_kl

        if not isinstance(num_epochs, (int, np.integer)) or num_epochs < 1:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite.")
        if not np.isfinite(beta_kl) or beta_kl < 0:
            raise ValueError("beta_kl must be non-negative and finite.")
        if not isinstance(kl_warmup_epochs, (int, np.integer)) or kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be a non-negative integer.")
        if max_grad_norm is not None and (
            not np.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive and finite or None.")
        if early_stopping_patience is not None and (
            not isinstance(early_stopping_patience, (int, np.integer))
            or early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer or None.")
        if not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative and finite.")

        X_train = self._validate_matrix(X_train, "X", self.x_dim)
        Y_train = self._validate_outcomes(Y_train, "Y_train")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X and Y_train must contain the same number of rows.")
        Y_mask_train = self._validate_categorical_mask(
            Y_mask_train, "Y_mask_train", X_train.shape[0]
        )

        if (X_val is None) != (Y_val is None):
            raise ValueError("X_val and Y_val must either both be provided or both be None.")
        if X_val is not None:
            X_val = self._validate_matrix(X_val, "X_val", self.x_dim)
            Y_val = self._validate_outcomes(Y_val, "Y_val")
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError("X_val and Y_val must contain the same number of rows.")
            Y_mask_val = self._validate_categorical_mask(
                Y_mask_val, "Y_mask_val", X_val.shape[0]
            )

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self._fit_standardizer(X_train)
        X_train_std = self._standardize(X_train)

        if X_val is not None:
            X_val_std = self._standardize(X_val)
        else:
            X_val_std = None

        train_ds = XYDataset(X_train_std, Y_train, mask=Y_mask_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if X_val_std is not None and Y_val is not None:
            val_ds = XYDataset(X_val_std, Y_val, mask=Y_mask_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_recon_per_outcome": [],
            "train_kl_loss": [],
            "train_kl_per_latent": [],
            "active_latent_units": [],
            "effective_beta_kl": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_recon_per_outcome": [],
            "val_kl_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_total = 0.0
            train_recon = 0.0
            train_kl = 0.0
            train_kl_by_latent = torch.zeros(self.latent_dim, device=self.device)
            train_rows = 0
            if kl_warmup_epochs > 0:
                effective_beta_kl = beta_kl * min(1.0, epoch / kl_warmup_epochs)
            else:
                effective_beta_kl = beta_kl

            for batch in train_loader:
                if len(batch) == 2:
                    xb, yb = batch
                    mb = None
                else:
                    xb, yb, mb = batch

                xb = xb.to(self.device)
                yb = yb.to(self.device)
                if mb is not None:
                    mb = mb.to(self.device)

                optimizer.zero_grad()
                out, mu_z, logvar_z = self.model(xb, yb, mask=mb)

                recon_loss = self._recon_loss(yb, out, mask=mb)
                kl_loss = -0.5 * torch.sum(
                    1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                )
                kl_by_latent = -0.5 * torch.sum(
                    1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=0
                )

                # Mild L2 penalty on NB raw parameters (to discourage huge mus/rs)
                penalty = torch.tensor(0.0, device=self.device)
                if self.outcome_type == "neg_binomial":
                    raw_mu = out["raw_mu"]
                    raw_r = out["raw_r"]
                    penalty = 1e-5 * (raw_mu.pow(2).mean() + raw_r.pow(2).mean())

                batch_sz = xb.size(0)
                loss = (
                    recon_loss + effective_beta_kl * kl_loss + penalty
                ) / batch_sz
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Training loss became non-finite; check inputs and hyperparameters."
                    )
                loss.backward()
                if max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Training gradients became non-finite.")
                optimizer.step()

                train_total += loss.item() * batch_sz
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
                train_kl_by_latent += kl_by_latent.detach()
                train_rows += batch_sz

            train_loss_epoch = train_total / train_rows
            train_recon_epoch = train_recon / train_rows
            train_kl_epoch = train_kl / train_rows
            history["train_loss"].append(train_loss_epoch)
            history["train_recon_loss"].append(train_recon_epoch)
            history["train_recon_per_outcome"].append(
                train_recon_epoch / self.y_dim
            )
            history["train_kl_loss"].append(train_kl_epoch)
            kl_per_latent = (train_kl_by_latent / train_rows).cpu().numpy()
            history["train_kl_per_latent"].append(kl_per_latent)
            history["active_latent_units"].append(int(np.sum(kl_per_latent > 0.01)))
            history["effective_beta_kl"].append(effective_beta_kl)

            val_loss_epoch = None
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                val_recon = 0.0
                val_kl = 0.0
                val_rows = 0
                cuda_devices = []
                if self.device.type == "cuda":
                    cuda_devices = [self.device.index or torch.cuda.current_device()]
                with torch.random.fork_rng(devices=cuda_devices):
                    validation_seed = 0 if seed is None else seed + 1_000_000
                    torch.manual_seed(validation_seed)
                    with torch.no_grad():
                        for batch in val_loader:
                            if len(batch) == 2:
                                xb, yb = batch
                                mb = None
                            else:
                                xb, yb, mb = batch

                            xb = xb.to(self.device)
                            yb = yb.to(self.device)
                            if mb is not None:
                                mb = mb.to(self.device)

                            out, mu_z, logvar_z = self.model(xb, yb, mask=mb)
                            recon_loss = self._recon_loss(yb, out, mask=mb)
                            kl_loss = -0.5 * torch.sum(
                                1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                            )
                            batch_sz = xb.size(0)
                            loss = (
                                recon_loss + effective_beta_kl * kl_loss
                            ) / batch_sz
                            val_total += loss.item() * batch_sz
                            val_recon += recon_loss.item()
                            val_kl += kl_loss.item()
                            val_rows += batch_sz

                    val_loss_epoch = val_total / val_rows
                    val_recon_epoch = val_recon / val_rows
                    val_kl_epoch = val_kl / val_rows
                history["val_loss"].append(val_loss_epoch)
                history["val_recon_loss"].append(val_recon_epoch)
                history["val_recon_per_outcome"].append(
                    val_recon_epoch / self.y_dim
                )
                history["val_kl_loss"].append(val_kl_epoch)

                if val_loss_epoch < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss_epoch
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if verbose:
                if val_loss_epoch is not None:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss: {train_loss_epoch:.4f} | "
                        f"val loss: {val_loss_epoch:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:03d} | train loss: {train_loss_epoch:.4f}")

            if (
                val_loader is not None
                and early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        history["best_epoch"] = best_epoch
        history["epochs_ran"] = len(history["train_loss"])

        self.trained = True
        return history

    @staticmethod
    def _inference_slices(n_rows: int, inference_batch_size: Optional[int]):
        if inference_batch_size is None:
            inference_batch_size = n_rows
        if (
            not isinstance(inference_batch_size, (int, np.integer))
            or inference_batch_size < 1
        ):
            raise ValueError("inference_batch_size must be a positive integer or None.")
        for start in range(0, n_rows, int(inference_batch_size)):
            yield slice(start, min(start + int(inference_batch_size), n_rows))

    def _forward_logits(self, X_std: np.ndarray) -> torch.Tensor:
        """
        Internal helper used in tests: given standardized X, produce a single
        draw of decoder logits for Bernoulli outcomes.
        """
        assert self.outcome_type == "bernoulli", (
            "_forward_logits is only meaningful for outcome_type='bernoulli'."
        )

        X_std = np.asarray(X_std, dtype=np.float32)
        n, x_dim = X_std.shape
        assert x_dim == self.x_dim

        x_tensor = torch.from_numpy(X_std).to(self.device)

        # Use a fixed z (zeros) for determinism and simplicity
        z = torch.zeros((n, self.latent_dim), device=self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model.decode(x_tensor, z)
            logits = out["logits"]
        return logits

    # --------- prediction: distribution parameters ---------
    def predict_params(
        self,
        X: np.ndarray,
        n_mc: int = 20,
        inference_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return predictive distribution parameters for Y | X.

        bernoulli:
            {"probs": p_ij}

        gaussian:
            {"mu": mu_pred_ij, "sigma": sigma_pred_ij}

        poisson:
            {"rate": lambda_pred_ij, "var_y": var_y_pred_ij}
              where var_y approximates Var(Y_ij | X_i) under the mixture.

        categorical:
            {"probabilities": {outcome_name: p_ijk}}

        neg_binomial:
            {"mu": mu_pred_ij, "var_y": var_y_pred_ij}
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        if not isinstance(n_mc, (int, np.integer)) or n_mc < 1:
            raise ValueError("n_mc must be a positive integer.")
        X = self._validate_matrix(X, "X", self.x_dim)
        X_std = self._standardize(X)
        n = X.shape[0]
        if self.outcome_type == "categorical":
            results = {
                "probabilities": {
                    schema_entry["name"]: np.empty(
                        (n, len(schema_entry["levels"])), dtype=np.float32
                    )
                    for schema_entry in self.outcome_schema
                }
            }
        else:
            result_keys = {
                "bernoulli": ("probs",),
                "gaussian": ("mu", "sigma"),
                "poisson": ("rate", "var_y"),
                "neg_binomial": ("mu", "var_y"),
            }
            results = {
                key: np.empty((n, self.y_dim), dtype=np.float32)
                for key in result_keys[self.outcome_type]
            }

        self.model.eval()
        with torch.no_grad():
            for row_slice in self._inference_slices(n, inference_batch_size):
                x_tensor = torch.from_numpy(X_std[row_slice]).to(self.device)
                batch_n = x_tensor.shape[0]

                if self.outcome_type == "bernoulli":
                    sum_probs = torch.zeros((batch_n, self.y_dim), device=self.device)
                    for _ in range(n_mc):
                        z = torch.randn((batch_n, self.latent_dim), device=self.device)
                        sum_probs += torch.sigmoid(self.model.decode(x_tensor, z)["logits"])
                    results["probs"][row_slice] = (sum_probs / float(n_mc)).cpu().numpy()

                elif self.outcome_type == "gaussian":
                    sum_mu = torch.zeros((batch_n, self.y_dim), device=self.device)
                    sum_y2 = torch.zeros((batch_n, self.y_dim), device=self.device)
                    for _ in range(n_mc):
                        out = self.model.decode(
                            x_tensor,
                            torch.randn((batch_n, self.latent_dim), device=self.device),
                        )
                        mu = out["mu"]
                        sum_mu += mu
                        sum_y2 += torch.exp(out["logvar"]) + mu.square()
                    mu_pred = sum_mu / float(n_mc)
                    var_pred = torch.clamp(
                        sum_y2 / float(n_mc) - mu_pred.square(), min=1e-8
                    )
                    results["mu"][row_slice] = mu_pred.cpu().numpy()
                    results["sigma"][row_slice] = torch.sqrt(var_pred).cpu().numpy()

                elif self.outcome_type == "poisson":
                    sum_rate = torch.zeros((batch_n, self.y_dim), device=self.device)
                    sum_rate_sq = torch.zeros((batch_n, self.y_dim), device=self.device)
                    for _ in range(n_mc):
                        out = self.model.decode(
                            x_tensor,
                            torch.randn((batch_n, self.latent_dim), device=self.device),
                        )
                        rate = torch.exp(out["log_rate"])
                        sum_rate += rate
                        sum_rate_sq += rate.square()
                    rate_mean = sum_rate / float(n_mc)
                    var_rate = torch.clamp(
                        sum_rate_sq / float(n_mc) - rate_mean.square(), min=0.0
                    )
                    results["rate"][row_slice] = rate_mean.cpu().numpy()
                    results["var_y"][row_slice] = (rate_mean + var_rate).cpu().numpy()

                elif self.outcome_type == "categorical":
                    probability_sums = [
                        torch.zeros(
                            (batch_n, stop - start), device=self.device
                        )
                        for start, stop in self.outcome_slices
                    ]
                    for _ in range(n_mc):
                        logits = self.model.decode(
                            x_tensor,
                            torch.randn(
                                (batch_n, self.latent_dim), device=self.device
                            ),
                        )["logits"]
                        for outcome_index, (start, stop) in enumerate(
                            self.outcome_slices
                        ):
                            probability_sums[outcome_index] += torch.softmax(
                                logits[:, start:stop], dim=1
                            )
                    for outcome_index, schema_entry in enumerate(
                        self.outcome_schema
                    ):
                        results["probabilities"][schema_entry["name"]][
                            row_slice
                        ] = (
                            probability_sums[outcome_index] / float(n_mc)
                        ).cpu().numpy()

                elif self.outcome_type == "neg_binomial":
                    sum_mu = torch.zeros((batch_n, self.y_dim), device=self.device)
                    sum_y2 = torch.zeros((batch_n, self.y_dim), device=self.device)
                    for _ in range(n_mc):
                        out = self.model.decode(
                            x_tensor,
                            torch.randn((batch_n, self.latent_dim), device=self.device),
                        )
                        mu = F.softplus(out["raw_mu"]) + 1e-8
                        r = F.softplus(out["raw_r"]) + 1e-8
                        sum_mu += mu
                        sum_y2 += mu + mu.square() / r + mu.square()
                    mu_pred = sum_mu / float(n_mc)
                    var_pred = torch.clamp(
                        sum_y2 / float(n_mc) - mu_pred.square(), min=1e-8
                    )
                    results["mu"][row_slice] = mu_pred.cpu().numpy()
                    results["var_y"][row_slice] = var_pred.cpu().numpy()

        return results

    # --------- prediction: mean / expectation ---------
    def predict_mean(
        self,
        X: np.ndarray,
        n_mc: int = 20,
        inference_batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict E[Y | X].

        - bernoulli: probabilities
        - gaussian: predictive mean
        - poisson:  predictive mean (rate)
        - neg_binomial: predictive mean
        """
        if self.outcome_type == "categorical":
            raise ValueError(
                "predict_mean() is not defined for nominal categorical outcomes; "
                "use predict_params() to obtain named level probabilities."
            )
        params = self.predict_params(
            X, n_mc=n_mc, inference_batch_size=inference_batch_size
        )
        if self.outcome_type == "bernoulli":
            return params["probs"]
        elif self.outcome_type == "gaussian":
            return params["mu"]
        elif self.outcome_type == "poisson":
            return params["rate"]
        elif self.outcome_type == "neg_binomial":
            return params["mu"]
        else:
            raise ValueError("Invalid outcome_type.")

    def predict_proba(
        self,
        X: np.ndarray,
        n_mc: int = 20,
        inference_batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        For backward compatibility: only valid for Bernoulli outcomes.

        For non-Bernoulli outcome types, use predict_mean() or predict_params().
        """
        if self.outcome_type != "bernoulli":
            raise ValueError(
                "predict_proba() is only defined for outcome_type='bernoulli'. "
                "Use predict_params() for categorical models or predict_mean() "
                "for other supported families."
            )
        params = self.predict_params(
            X, n_mc=n_mc, inference_batch_size=inference_batch_size
        )
        return params["probs"]

    def evaluate_marginal_log_score(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_mc: int = 20,
        eps: float = 1e-7,
        Y_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Marginal composite log score for Bernoulli outcomes.

        This averages each outcome probability over the latent prior and then
        scores outcomes independently. It is not the joint log-likelihood of
        the full outcome vector.
        """
        if self.outcome_type != "bernoulli":
            raise NotImplementedError(
                "evaluate_loglik is currently implemented only for "
                "outcome_type='bernoulli'."
            )

        params = self.predict_params(X, n_mc=n_mc)
        probs = params["probs"]
        Y = np.asarray(Y, dtype=np.float32)
        assert Y.shape == probs.shape

        p = np.clip(probs, eps, 1.0 - eps)
        ll_matrix = Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)

        if Y_mask is not None:
            mask = np.asarray(Y_mask, dtype=np.float32)
            assert mask.shape == Y.shape
            ll_matrix = ll_matrix * mask
            denom = float(mask.sum())
        else:
            denom = float(Y.shape[0] * Y.shape[1])

        sum_ll = float(ll_matrix.sum())
        avg_ll = float(sum_ll / max(denom, eps))
        avg_bce = float(-avg_ll)

        return {
            "sum_loglik": sum_ll,
            "avg_loglik": avg_ll,
            "avg_bce": avg_bce,
        }

    def evaluate_loglik(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_mc: int = 20,
        eps: float = 1e-7,
        Y_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Deprecated alias for :meth:`evaluate_marginal_log_score`."""
        warnings.warn(
            "evaluate_loglik() computes a marginal composite score, not a joint "
            "log-likelihood; use evaluate_marginal_log_score().",
            FutureWarning,
            stacklevel=2,
        )
        return self.evaluate_marginal_log_score(
            X=X, Y=Y, n_mc=n_mc, eps=eps, Y_mask=Y_mask
        )

    # --------- generation ---------
    def generate(
        self,
        X_new: np.ndarray,
        n_samples_per_x: int = 1,
        return_probs: bool = False,
        inference_batch_size: Optional[int] = None,
        decoder_batch_size: int = 10000,
    ) -> Any:
        """
        Generate samples from p(Y | X).

        - bernoulli:
            - return_probs=True : probabilities (via MC mean)
            - return_probs=False: Bernoulli samples
        - gaussian:
            - return_probs=True : predictive means
            - return_probs=False: Normal samples
        - poisson:
            - return_probs=True : predictive rates (lambda)
            - return_probs=False: Poisson samples
        - neg_binomial:
            - return_probs=True : predictive means
            - return_probs=False: NB samples
        - categorical:
            - return_probs=True : named level-probability matrices
            - return_probs=False: zero-based integer-coded samples
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before generation.")
        if not isinstance(n_samples_per_x, (int, np.integer)) or n_samples_per_x < 1:
            raise ValueError("n_samples_per_x must be a positive integer.")
        if self.outcome_type == "categorical" and (
            not isinstance(decoder_batch_size, (int, np.integer))
            or decoder_batch_size < 1
        ):
            raise ValueError("decoder_batch_size must be a positive integer.")

        X_new = self._validate_matrix(X_new, "X", self.x_dim)
        X_std = self._standardize(X_new)
        n, x_dim = X_std.shape

        # When return_probs=True, use predict_params() to get distribution params
        if return_probs:
            params = self.predict_params(
                X_new,
                n_mc=max(n_samples_per_x, 10),
                inference_batch_size=inference_batch_size,
            )
            if self.outcome_type == "bernoulli":
                return params["probs"]
            elif self.outcome_type == "gaussian":
                return params["mu"]
            elif self.outcome_type == "poisson":
                return params["rate"]
            elif self.outcome_type == "categorical":
                return params["probabilities"]
            elif self.outcome_type == "neg_binomial":
                return params["mu"]
            else:
                raise ValueError("Invalid outcome_type.")

        if n_samples_per_x == 1:
            output_shape = (n, self.y_dim)
        else:
            output_shape = (n, n_samples_per_x, self.y_dim)
        output_dtype = np.float32 if self.outcome_type == "gaussian" else np.int32
        samples = np.empty(output_shape, dtype=output_dtype)

        self.model.eval()
        with torch.no_grad():
            for row_slice in self._inference_slices(n, inference_batch_size):
                x_tensor = torch.from_numpy(X_std[row_slice]).to(self.device)
                batch_n = x_tensor.shape[0]
                total = batch_n * n_samples_per_x

                if self.outcome_type == "categorical":
                    # Write each bounded decoder chunk directly into the final
                    # output buffer. This avoids materializing expanded X/Z/logit
                    # tensors or a second full-size sample array.
                    batch_samples = samples[row_slice].reshape(total, self.y_dim)
                    for expanded_start in range(
                        0, total, int(decoder_batch_size)
                    ):
                        expanded_stop = min(
                            expanded_start + int(decoder_batch_size), total
                        )
                        row_indices = torch.div(
                            torch.arange(
                                expanded_start,
                                expanded_stop,
                                device=self.device,
                            ),
                            n_samples_per_x,
                            rounding_mode="floor",
                        )
                        x_chunk = x_tensor.index_select(0, row_indices)
                        z = torch.randn(
                            (expanded_stop - expanded_start, self.latent_dim),
                            device=self.device,
                        )
                        logits = self.model.decode(x_chunk, z)["logits"]
                        for outcome_index, (start, stop) in enumerate(
                            self.outcome_slices
                        ):
                            batch_samples[
                                expanded_start:expanded_stop, outcome_index
                            ] = (
                                torch.distributions.Categorical(
                                    logits=logits[:, start:stop]
                                )
                                .sample()
                                .cpu()
                                .numpy()
                                .astype(np.int32)
                            )
                    continue

                x_rep = x_tensor.repeat_interleave(n_samples_per_x, dim=0)
                z = torch.randn((total, self.latent_dim), device=self.device)
                out = self.model.decode(x_rep, z)

                if self.outcome_type == "bernoulli":
                    distribution = torch.distributions.Bernoulli(
                        probs=torch.sigmoid(out["logits"])
                    )
                elif self.outcome_type == "gaussian":
                    distribution = torch.distributions.Normal(
                        loc=out["mu"], scale=torch.exp(0.5 * out["logvar"])
                    )
                elif self.outcome_type == "poisson":
                    distribution = torch.distributions.Poisson(torch.exp(out["log_rate"]))
                elif self.outcome_type == "neg_binomial":
                    mu = F.softplus(out["raw_mu"]) + 1e-8
                    r = F.softplus(out["raw_r"]) + 1e-8
                    p = torch.clamp(r / (r + mu), min=1e-6, max=1.0 - 1e-6)
                    distribution = torch.distributions.NegativeBinomial(
                        total_count=r, probs=p
                    )
                else:
                    raise ValueError("Invalid outcome_type.")

                batch_samples = distribution.sample().cpu().numpy().astype(output_dtype)
                if n_samples_per_x == 1:
                    samples[row_slice] = batch_samples.reshape(batch_n, self.y_dim)
                else:
                    samples[row_slice] = batch_samples.reshape(
                        batch_n, n_samples_per_x, self.y_dim
                    )

        return samples


# ---------------------------------------------------------------------------
# Tuning helpers (unchanged from your last version)
# ---------------------------------------------------------------------------


def tune_cvae_random_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    x_dim: int,
    y_dim: int,
    search_space: Dict[str, List[Any]],
    n_trials: int = 20,
    outcome_type: str = "bernoulli",
    device: Optional[str] = None,
    base_seed: int = 1234,
    verbose: bool = True,
    outcome_schema: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample_config() -> Dict[str, Any]:
        cfg = {}
        for k, vals in search_space.items():
            cfg[k] = random.choice(vals)
        return cfg

    trials: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_config: Optional[Dict[str, Any]] = None

    for t in range(n_trials):
        cfg = sample_config()

        trainer = CVAETrainer(
            x_dim=x_dim,
            y_dim=y_dim,
            latent_dim=cfg.get("latent_dim", 8),
            outcome_type=outcome_type,
            enc_hidden_dims=cfg.get("enc_hidden_dims", None),
            dec_hidden_dims=cfg.get("dec_hidden_dims", None),
            hidden_dim=cfg.get("hidden_dim", 64),
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            num_epochs=cfg.get("num_epochs", 50),
            batch_size=cfg.get("batch_size", 256),
            lr=cfg.get("lr", 1e-3),
            beta_kl=cfg.get("beta_kl", 1.0),
            device=device,
            outcome_schema=outcome_schema,
        )

        if verbose:
            print(f"\n=== Random Trial {t+1}/{n_trials} ===")
            print("Config:", cfg)

        history = trainer.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            num_epochs=cfg.get("num_epochs", None),
            batch_size=cfg.get("batch_size", None),
            lr=cfg.get("lr", None),
            beta_kl=cfg.get("beta_kl", None),
            verbose=verbose,
            seed=base_seed + t,
        )

        if len(history.get("val_loss", [])) > 0:
            val_loss = history["val_loss"][-1]
        else:
            val_loss = history["train_loss"][-1]

        trials.append({"config": cfg, "val_loss": val_loss})

        if verbose:
            print(f"Random trial {t+1} val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = cfg

    return {
        "trials": trials,
        "best_config": best_config,
        "best_val_loss": best_val_loss,
    }


def tune_cvae_tpe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    x_dim: int,
    y_dim: int,
    search_space: Dict[str, List[Any]],
    n_trials: int = 20,
    outcome_type: str = "bernoulli",
    device: Optional[str] = None,
    base_seed: int = 1234,
    verbose: bool = True,
    outcome_schema: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    except ImportError as e:
        raise ImportError(
            "hyperopt is required for tune_cvae_tpe. "
            "Install via `pip install hyperopt`."
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hp_space = {k: hp.choice(k, vals) for k, vals in search_space.items()}
    trials_hpo = Trials()

    def objective(cfg: Dict[str, Any]):
        nonlocal base_seed

        trainer = CVAETrainer(
            x_dim=x_dim,
            y_dim=y_dim,
            latent_dim=cfg.get("latent_dim", 8),
            outcome_type=outcome_type,
            enc_hidden_dims=cfg.get("enc_hidden_dims", None),
            dec_hidden_dims=cfg.get("dec_hidden_dims", None),
            hidden_dim=cfg.get("hidden_dim", 64),
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            num_epochs=cfg.get("num_epochs", 50),
            batch_size=cfg.get("batch_size", 256),
            lr=cfg.get("lr", 1e-3),
            beta_kl=cfg.get("beta_kl", 1.0),
            device=device,
            outcome_schema=outcome_schema,
        )

        if verbose:
            print("\n=== TPE trial ===")
            print("Config:", cfg)

        history = trainer.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            num_epochs=cfg.get("num_epochs", None),
            batch_size=cfg.get("batch_size", None),
            lr=cfg.get("lr", None),
            beta_kl=cfg.get("beta_kl", None),
            verbose=verbose,
            seed=base_seed,
        )

        if len(history.get("val_loss", [])) > 0:
            val_loss = history["val_loss"][-1]
        else:
            val_loss = history["train_loss"][-1]

        if verbose:
            print(f"TPE trial val_loss: {val_loss:.4f}")

        return {"loss": val_loss, "status": STATUS_OK, "config": cfg}

    fmin(
        fn=objective,
        space=hp_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials_hpo,
        rstate=np.random.default_rng(base_seed),
    )

    trials: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_config: Optional[Dict[str, Any]] = None

    for tr in trials_hpo.trials:
        result = tr["result"]
        cfg = result["config"]
        loss = float(result["loss"])
        trials.append({"config": cfg, "val_loss": loss})
        if loss < best_val_loss:
            best_val_loss = loss
            best_config = cfg

    return {
        "trials": trials,
        "best_config": best_config,
        "best_val_loss": best_val_loss,
    }


def fit_cvae_with_tuning(
    X: np.ndarray,
    Y: np.ndarray,
    search_space: Dict[str, List[Any]],
    method: str = "random",
    n_trials: int = 20,
    train_frac: float = 0.8,
    outcome_type: str = "bernoulli",
    seed: int = 1234,
    device: Optional[str] = None,
    verbose: bool = True,
    outcome_schema: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    One-shot convenience wrapper:

      1. Split X, Y into train/val
      2. Tune hyperparameters
      3. Refit on full dataset with best config
      4. Return fitted trainer + tuning info
    """
    X = np.asarray(X, dtype=np.float32)
    # Preserve categorical values until the trainer has verified that the
    # original inputs are exact integer codes. Numeric families retain the
    # historical float32 conversion.
    if outcome_type == "categorical":
        Y = np.asarray(Y)
    else:
        Y = np.asarray(Y, dtype=np.float32)
    assert X.shape[0] == Y.shape[0]
    n, x_dim = X.shape
    y_dim = Y.shape[1]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]

    if method.lower() == "random":
        tuning_results = tune_cvae_random_search(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            x_dim=x_dim,
            y_dim=y_dim,
            search_space=search_space,
            n_trials=n_trials,
            outcome_type=outcome_type,
            device=device,
            base_seed=seed,
            verbose=verbose,
            outcome_schema=outcome_schema,
        )
    elif method.lower() == "tpe":
        tuning_results = tune_cvae_tpe(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            x_dim=x_dim,
            y_dim=y_dim,
            search_space=search_space,
            n_trials=n_trials,
            outcome_type=outcome_type,
            device=device,
            base_seed=seed,
            verbose=verbose,
            outcome_schema=outcome_schema,
        )
    else:
        raise ValueError("method must be 'random' or 'tpe'.")

    best_config = tuning_results["best_config"]
    if verbose:
        print("\nBest config from tuning:")
        print(best_config)

    trainer = CVAETrainer(
        x_dim=x_dim,
        y_dim=y_dim,
        latent_dim=best_config.get("latent_dim", 8),
        outcome_type=outcome_type,
        enc_hidden_dims=best_config.get("enc_hidden_dims", None),
        dec_hidden_dims=best_config.get("dec_hidden_dims", None),
        hidden_dim=best_config.get("hidden_dim", 64),
        n_hidden_layers=best_config.get("n_hidden_layers", 2),
        num_epochs=best_config.get("num_epochs", 50),
        batch_size=best_config.get("batch_size", 256),
        lr=best_config.get("lr", 1e-3),
        beta_kl=best_config.get("beta_kl", 1.0),
        device=device,
        outcome_schema=outcome_schema,
    )

    trainer.fit(
        X_train=X,
        Y_train=Y,
        X_val=None,
        Y_val=None,
        num_epochs=best_config.get("num_epochs", None),
        batch_size=best_config.get("batch_size", None),
        lr=best_config.get("lr", None),
        beta_kl=best_config.get("beta_kl", None),
        verbose=verbose,
        seed=seed,
    )

    return {
        "trainer": trainer,
        "best_config": best_config,
        "tuning_results": tuning_results,
        "train_indices": idx_train,
        "val_indices": idx_val,
    }
