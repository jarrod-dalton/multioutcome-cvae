import numpy as np
from typing import Optional, Dict, Any, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


VALID_OUTCOME_TYPES = ("bernoulli", "gaussian", "poisson", "neg_binomial")


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

      outcome_type ∈ {"bernoulli", "gaussian", "poisson"}

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
    ):
        super().__init__()
        assert outcome_type in VALID_OUTCOME_TYPES, \
            f"outcome_type must be one of {VALID_OUTCOME_TYPES}"
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.outcome_type = outcome_type

        if enc_hidden_dims is None or len(enc_hidden_dims) == 0:
            enc_hidden_dims = [64, 64]
        if dec_hidden_dims is None or len(dec_hidden_dims) == 0:
            dec_hidden_dims = enc_hidden_dims

        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims

        # ----- Encoder over [x, y] -----
        enc_input_dim = x_dim + y_dim
        self.enc_layers = nn.ModuleList()
        in_dim = enc_input_dim
        for h_dim in enc_hidden_dims:
            self.enc_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        enc_last_dim = in_dim

        self.enc_mu = nn.Linear(enc_last_dim, latent_dim)
        self.enc_logvar = nn.Linear(enc_last_dim, latent_dim)

        # ----- Decoder over [x, z] -----
        dec_input_dim = x_dim + latent_dim
        self.dec_layers = nn.ModuleList()
        in_dim = dec_input_dim
        for h_dim in dec_hidden_dims:
            self.dec_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        dec_last_dim = in_dim

        if outcome_type == "bernoulli":
            self.dec_out = nn.Linear(dec_last_dim, y_dim)  # logits
        elif outcome_type == "gaussian":
            self.dec_mu = nn.Linear(dec_last_dim, y_dim)
            self.dec_logvar = nn.Linear(dec_last_dim, y_dim)
        elif outcome_type == "poisson":
            self.dec_log_rate = nn.Linear(dec_last_dim, y_dim)
        elif outcome_type == "neg_binomial":
            self.dec_log_mu = nn.Linear(dec_last_dim, y_dim)
            self.dec_log_r = nn.Linear(dec_last_dim, y_dim)

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        h = torch.cat([x, y], dim=1)
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
        h = torch.cat([x, z], dim=1)
        for layer in self.dec_layers:
            h = F.relu(layer(h))

        if self.outcome_type == "bernoulli":
            logits = self.dec_out(h)
            return {"logits": logits}
        elif self.outcome_type == "gaussian":
            mu = self.dec_mu(h)
            logvar = self.dec_logvar(h)
            return {"mu": mu, "logvar": logvar}
        elif self.outcome_type == "poisson":
            log_rate = self.dec_log_rate(h)
            return {"log_rate": log_rate}
        elif self.outcome_type == "neg_binomial":
            log_mu = self.dec_log_mu(h)
            log_r  = self.dec_log_r(h)
            return {"log_mu": log_mu, "log_r": log_r}
        else:
            raise ValueError("Invalid outcome_type.")

    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Standard forward pass: encode -> reparameterize -> decode.
        Returns:
            out: decoder outputs (family-specific)
            mu_z, logvar_z: parameters of q(z | x, y)
        """
        mu_z, logvar_z = self.encode(x, y)
        z = self.reparameterize(mu_z, logvar_z)
        out = self.decode(x, z)
        return out, mu_z, logvar_z


# Example usage:
#
# from multioutcome_cvae import CVAETrainer
#
# trainer = CVAETrainer(
#     x_dim=5,
#     y_dim=10,
#     latent_dim=8,
#     outcome_type="bernoulli",
#     hidden_dim=64,
#     n_hidden_layers=2,
# )
# trainer.fit(X_train, Y_train)
#
class CVAETrainer:
    """
    CVAETrainer

    High-level wrapper around MultivariateOutcomeCVAE. Handles:

    - construction of encoder/decoder neural networks
    - standardization of X
    - training loop (reconstruction + KL)
    - prediction and generation helpers
    - outcome-family-specific behavior

    Parameters
    ----------
    (docstring unchanged; see earlier version)
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
    ):
        assert outcome_type in VALID_OUTCOME_TYPES, \
            f"outcome_type must be one of {VALID_OUTCOME_TYPES}"
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.outcome_type = outcome_type

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
        ).to(self.device)

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.trained: bool = False

    # --------- standardization helpers ---------
    def _fit_standardizer(self, X_train: np.ndarray):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        self.x_mean = mean.astype(np.float32)
        self.x_std = std.astype(np.float32)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize X using stored mean/std. If the standardizer has not
        been fit yet (x_mean/x_std is None), fit it on the provided X.

        This makes it safe to call _standardize() before .fit(), which
        is convenient in tests and quick exploratory use.
        """
        X = np.asarray(X, dtype=np.float32)
        if self.x_mean is None or self.x_std is None:
            self._fit_standardizer(X)
        return (X - self.x_mean) / self.x_std

    # --------- negative binomial: -log likelihood ---------
    import math
    from torch.special import gammaln  # PyTorch >= 1.8
    
    def _neg_binomial_nll(y: torch.Tensor,
                          log_mu: torch.Tensor,
                          log_r: torch.Tensor,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Negative binomial negative log-likelihood (sum over all elements).
    
        Parameterization:
          mean mu > 0, dispersion r > 0
          Var(Y) = mu + mu^2 / r
    
        We use:
          Y ~ NB(r, p),  with
            mean = r * (1 - p) / p = mu
            p = r / (r + mu)
    
        log p(Y = y) =
            lgamma(y + r) - lgamma(r) - lgamma(y + 1)
            + r * log(r / (r + mu))
            + y * log(mu / (r + mu))
        """
        mu = torch.exp(log_mu)
        r  = torch.exp(log_r)
    
        # Avoid degeneracies
        mu = torch.clamp(mu, min=1e-8)
        r  = torch.clamp(r,  min=1e-8)
    
        # log p(y)
        t1 = gammaln(y + r) - gammaln(r)  # - gammaln(y+1) is constant in params; can omit
        log_r_over_rplusmu = torch.log(r) - torch.log(r + mu)
        log_mu_over_rplusmu = torch.log(mu) - torch.log(r + mu)
    
        t2 = r * log_r_over_rplusmu
        t3 = y * log_mu_over_rplusmu
    
        log_p = t1 + t2 + t3
    
        if mask is not None:
            log_p = log_p * mask
    
        # Negative log-likelihood (sum)
        return -torch.sum(log_p)
    
    # --------- reconstruction loss by outcome family ---------
    def _recon_loss(
        self,
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
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
    
        elif self.outcome_type == "neg_binomial":
            log_mu = out["log_mu"]
            log_r  = out["log_r"]
            return _neg_binomial_nll(y, log_mu, log_r, mask=mask)
    
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

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

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

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss_epoch = 0.0
            n_batches = 0

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
                out, mu_z, logvar_z = self.model(xb, yb)

                recon_loss = self._recon_loss(yb, out, mask=mb)
                kl_loss = -0.5 * torch.sum(
                    1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                )

                batch_sz = xb.size(0)
                loss = (recon_loss + beta_kl * kl_loss) / batch_sz
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                n_batches += 1

            train_loss_epoch /= max(1, n_batches)
            history["train_loss"].append(train_loss_epoch)

            val_loss_epoch = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
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

                        out, mu_z, logvar_z = self.model(xb, yb)
                        recon_loss = self._recon_loss(yb, out, mask=mb)
                        kl_loss = -0.5 * torch.sum(
                            1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                        )
                        batch_sz = xb.size(0)
                        loss = (recon_loss + beta_kl * kl_loss) / batch_sz
                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss_epoch = val_loss / max(1, n_val_batches)
                history["val_loss"].append(val_loss_epoch)

            if verbose:
                if val_loss_epoch is not None:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss: {train_loss_epoch:.4f} | "
                        f"val loss: {val_loss_epoch:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:03d} | train loss: {train_loss_epoch:.4f}")

        self.trained = True
        return history

    # --------- internal helper for MC over z ---------
    def _decode_for_X(self, X_std: np.ndarray, n_mc: int):
        """Internal helper: Monte Carlo over z, returns list of decoded outputs."""
        n = X_std.shape[0]
        x_tensor = torch.from_numpy(X_std).to(self.device)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_mc):
                z = torch.randn((n, self.latent_dim), device=self.device)
                out = self.model.decode(x_tensor, z)
                outs.append(out)
        return outs

    def _forward_logits(self, X_std: np.ndarray) -> torch.Tensor:
        """
        Internal helper used in tests: given standardized X, produce a single
        draw of decoder logits for Bernoulli outcomes.

        Parameters
        ----------
        X_std : np.ndarray, shape (n, x_dim)
            Standardized covariate matrix.

        Returns
        -------
        torch.Tensor, shape (n, y_dim)
            Decoder logits for Y | X, Z with a fixed Z sample.
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
    ) -> Dict[str, np.ndarray]:
        """
        Return predictive distribution parameters for Y | X.

        (docstring unchanged)
        """
        assert self.trained, "Model must be trained first."
        X = np.asarray(X, dtype=np.float32)
        X_std = self._standardize(X)
        n = X.shape[0]

        outs = self._decode_for_X(X_std, n_mc=n_mc)

        if self.outcome_type == "bernoulli":
            acc = torch.zeros((n, self.y_dim), device=self.device)
            for out in outs:
                logits = out["logits"]
                probs = torch.sigmoid(logits)
                acc += probs
            probs = (acc / float(n_mc)).cpu().numpy()
            return {"probs": probs}

        elif self.outcome_type == "gaussian":
            sum_mu = torch.zeros((n, self.y_dim), device=self.device)
            sum_m2_plus_var = torch.zeros((n, self.y_dim), device=self.device)

            for out in outs:
                mu = out["mu"]
                logvar = out["logvar"]
                var = torch.exp(logvar)
                sum_mu += mu
                sum_m2_plus_var += var + mu ** 2

            mu_pred = sum_mu / float(n_mc)
            Ey2 = sum_m2_plus_var / float(n_mc)
            var_pred = Ey2 - mu_pred ** 2
            var_pred = torch.clamp(var_pred, min=1e-8)
            sigma_pred = torch.sqrt(var_pred)

            return {
                "mu": mu_pred.cpu().numpy(),
                "sigma": sigma_pred.cpu().numpy(),
            }

        elif self.outcome_type == "poisson":
            sum_rate = torch.zeros((n, self.y_dim), device=self.device)
            sum_rate_sq = torch.zeros((n, self.y_dim), device=self.device)

            for out in outs:
                log_rate = out["log_rate"]
                rate = torch.exp(log_rate)
                sum_rate += rate
                sum_rate_sq += rate ** 2

            lambda_mean = sum_rate / float(n_mc)
            Ey_lambda2 = sum_rate_sq / float(n_mc)
            var_lambda = Ey_lambda2 - lambda_mean ** 2
            var_lambda = torch.clamp(var_lambda, min=0.0)
            var_y = lambda_mean + var_lambda

            return {
                "rate": lambda_mean.cpu().numpy(),
                "var_y": var_y.cpu().numpy(),
            }

        elif self.outcome_type == "neg_binomial":
            # mixture of Negative Binomials; we track mean and Var(Y)
            sum_mu = torch.zeros((n, self.y_dim), device=self.device)
            sum_Ey2 = torch.zeros((n, self.y_dim), device=self.device)
        
            for out in outs:
                log_mu = out["log_mu"]
                log_r  = out["log_r"]
                mu = torch.exp(log_mu)
                r  = torch.exp(log_r)
        
                var_y = mu + mu**2 / r
                sum_mu  += mu
                sum_Ey2 += var_y + mu**2  # E[Y^2] = Var(Y) + (E[Y])^2
        
            mu_pred = sum_mu / float(n_mc)
            Ey2 = sum_Ey2 / float(n_mc)
            var_pred = Ey2 - mu_pred**2
            var_pred = torch.clamp(var_pred, min=1e-8)
        
            return {
                "mu": mu_pred.cpu().numpy(),
                "var_y": var_pred.cpu().numpy(),
            }

        else:
            raise ValueError("Invalid outcome_type.")

    # --------- prediction: mean / expectation ---------
    def predict_mean(
        self,
        X: np.ndarray,
        n_mc: int = 20,
    ) -> np.ndarray:
        params = self.predict_params(X, n_mc=n_mc)
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
    ) -> np.ndarray:
        """
        For backward compatibility: only valid for Bernoulli outcomes.
        """
        if self.outcome_type != "bernoulli":
            raise ValueError(
                "predict_proba() is only defined for outcome_type='bernoulli'. "
                "Use predict_mean() or predict_params() for gaussian/poisson."
            )
        params = self.predict_params(X, n_mc=n_mc)
        return params["probs"]

    def evaluate_loglik(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_mc: int = 20,
        eps: float = 1e-7,
        Y_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Log-likelihood-style evaluation for Bernoulli outcomes.

        For outcome_type != 'bernoulli', this is currently not implemented.

        Parameters
        ----------
        X : np.ndarray, shape (n, x_dim)
        Y : np.ndarray, shape (n, y_dim)
        n_mc : int
            Monte Carlo samples over z.
        eps : float
            Numerical epsilon to avoid log(0).
        Y_mask : np.ndarray or None, shape (n, y_dim)
            Optional mask over Y: 1 = observed, 0 = missing. If None,
            all entries are treated as observed.
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

    # --------- generation ---------
    def generate(
        self,
        X_new: np.ndarray,
        n_samples_per_x: int = 1,
        return_probs: bool = False,
    ) -> np.ndarray:
        """
        Generate samples from p(Y | X).

        (docstring unchanged)
        """
        assert self.trained, "Model must be trained first."

        X_new = np.asarray(X_new, dtype=np.float32)
        X_std = self._standardize(X_new)
        n, x_dim = X_std.shape
        assert x_dim == self.x_dim

        # When return_probs=True, use predict_params() to get distribution params
        if return_probs:
            params = self.predict_params(X_new, n_mc=max(n_samples_per_x, 10))
            if self.outcome_type == "bernoulli":
                return params["probs"]
            elif self.outcome_type == "gaussian":
                return params["mu"]
            elif self.outcome_type == "poisson":
                return params["rate"]
            else:
                raise ValueError("Invalid outcome_type.")

        # return_probs=False: sample from predictive distribution (1 draw per z)
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_std).to(self.device)

            total = n * n_samples_per_x
            x_rep = x_tensor.repeat_interleave(n_samples_per_x, dim=0)
            z = torch.randn((total, self.latent_dim), device=self.device)
            out = self.model.decode(x_rep, z)

            if self.outcome_type == "bernoulli":
                logits = out["logits"]
                probs = torch.sigmoid(logits)
                bern = torch.distributions.Bernoulli(probs=probs)
                y_samples = bern.sample().cpu().numpy().astype(np.int32)

            elif self.outcome_type == "gaussian":
                mu = out["mu"]
                logvar = out["logvar"]
                std = torch.exp(0.5 * logvar)
                normal = torch.distributions.Normal(loc=mu, scale=std)
                y_samples = normal.sample().cpu().numpy().astype(np.float32)

            elif self.outcome_type == "poisson":
                log_rate = out["log_rate"]
                rate = torch.exp(log_rate)
                pois = torch.distributions.Poisson(rate)
                y_samples = pois.sample().cpu().numpy().astype(np.int32)
                
            elif self.outcome_type == "neg_binomial":
                log_mu = out["log_mu"]
                log_r  = out["log_r"]
                mu = torch.exp(log_mu)
                r  = torch.exp(log_r)
            
                # NB parameterization: total_count = r, probability p = r / (r + mu)
                p = r / (r + mu)
                p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
            
                nb = torch.distributions.NegativeBinomial(total_count=r, probs=p)
                y_samples = nb.sample().cpu().numpy().astype(np.int32)

            else:
                raise ValueError("Invalid outcome_type.")

            if n_samples_per_x == 1:
                return y_samples.reshape(n, self.y_dim)
            else:
                return y_samples.reshape(n, n_samples_per_x, self.y_dim)


# --------- tuning helpers ---------


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
) -> Dict[str, Any]:
    """
    One-shot convenience wrapper:

      1. Split X, Y into train/val
      2. Tune hyperparameters
      3. Refit on full dataset with best config
      4. Return fitted trainer + tuning info
    """
    X = np.asarray(X, dtype=np.float32)
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
