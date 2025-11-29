import numpy as np
from typing import Optional, Dict, Any, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class XYDataset(Dataset):
    """Simple dataset wrapper for (X, Y) pairs."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultivariateBinaryCVAE(nn.Module):
    """Conditional VAE for multivariate binary outcomes.

    Encoder: q(z | x, y)
    Decoder: p(y | x, z)
    Prior:   p(z) = N(0, I)

    Now supports flexible hidden layer configurations via
    enc_hidden_dims and dec_hidden_dims.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int = 8,
        enc_hidden_dims: Optional[List[int]] = None,
        dec_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        # Defaults: two hidden layers of size 64, like before
        if enc_hidden_dims is None or len(enc_hidden_dims) == 0:
            enc_hidden_dims = [64, 64]
        if dec_hidden_dims is None or len(dec_hidden_dims) == 0:
            dec_hidden_dims = enc_hidden_dims

        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims

        # ----- Encoder: [X, Y] -> hidden -> (mu, logvar) -----
        enc_input_dim = x_dim + y_dim
        self.enc_layers = nn.ModuleList()
        in_dim = enc_input_dim
        for h_dim in enc_hidden_dims:
            self.enc_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        enc_last_dim = in_dim

        self.enc_mu = nn.Linear(enc_last_dim, latent_dim)
        self.enc_logvar = nn.Linear(enc_last_dim, latent_dim)

        # ----- Decoder: [X, z] -> hidden -> logits(Y) -----
        dec_input_dim = x_dim + latent_dim
        self.dec_layers = nn.ModuleList()
        in_dim = dec_input_dim
        for h_dim in dec_hidden_dims:
            self.dec_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        dec_last_dim = in_dim

        self.dec_out = nn.Linear(dec_last_dim, y_dim)  # logits

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        """q(z | x, y) parameterized by MLP over [x, y]."""
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

    def decode(self, x: torch.Tensor, z: torch.Tensor):
        """p(y | x, z) parameterized by MLP over [x, z]."""
        h = torch.cat([x, z], dim=1)
        for layer in self.dec_layers:
            h = F.relu(layer(h))
        logits = self.dec_out(h)
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(x, z)
        return logits, mu, logvar


class CVAETrainer:
    """High-level wrapper for training and using MultivariateBinaryCVAE.

    Designed to be easy to call from R via reticulate, and now supports
    flexible hidden configs:

    - Use enc_hidden_dims / dec_hidden_dims explicitly, OR
    - Use hidden_dim + n_hidden_layers for simple symmetric MLPs.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int = 8,
        # Flexible hidden config
        enc_hidden_dims: Optional[List[int]] = None,
        dec_hidden_dims: Optional[List[int]] = None,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        # default training hyperparams
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        device: Optional[str] = None,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        # Resolve hidden layer configs
        if enc_hidden_dims is None:
            enc_hidden_dims = [hidden_dim] * n_hidden_layers
        if dec_hidden_dims is None:
            dec_hidden_dims = enc_hidden_dims

        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims

        # Default training params (can override in fit())
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta_kl = beta_kl

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = MultivariateBinaryCVAE(
            x_dim=x_dim,
            y_dim=y_dim,
            latent_dim=latent_dim,
            enc_hidden_dims=enc_hidden_dims,
            dec_hidden_dims=dec_hidden_dims,
        ).to(self.device)

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.trained: bool = False

    # ---- internal: standardization ----
    def _fit_standardizer(self, X_train: np.ndarray):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div-by-zero
        self.x_mean = mean.astype(np.float32)
        self.x_std = std.astype(np.float32)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    # ---- training loop ----
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
    ) -> Dict[str, Any]:
        """Train the CVAE on (X_train, Y_train)."""

        # Resolve hyperparams (either passed-in or defaults)
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        lr = lr if lr is not None else self.lr
        beta_kl = beta_kl if beta_kl is not None else self.beta_kl

        # Reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # 1. Fit and apply standardization
        self._fit_standardizer(X_train)
        X_train_std = self._standardize(X_train)

        if X_val is not None:
            X_val_std = self._standardize(X_val)
        else:
            X_val_std = None

        # 2. Data loaders
        train_ds = XYDataset(X_train_std, Y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if X_val_std is not None:
            val_ds = XYDataset(X_val_std, Y_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss_epoch = 0.0
            n_train_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits, mu, logvar = self.model(xb, yb)

                # Reconstruction (binary cross-entropy)
                recon_loss = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="sum"
                )
                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                batch_size_curr = xb.size(0)
                loss = (recon_loss + beta_kl * kl_loss) / batch_size_curr
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                n_train_batches += 1

            train_loss_epoch /= max(1, n_train_batches)
            history["train_loss"].append(train_loss_epoch)

            # Validation
            val_loss_epoch = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        logits, mu, logvar = self.model(xb, yb)
                        recon_loss = F.binary_cross_entropy_with_logits(
                            logits, yb, reduction="sum"
                        )
                        kl_loss = -0.5 * torch.sum(
                            1 + logvar - mu.pow(2) - logvar.exp()
                        )
                        batch_size_curr = xb.size(0)
                        loss = (recon_loss + beta_kl * kl_loss) / batch_size_curr
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
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss: {train_loss_epoch:.4f}"
                    )

        self.trained = True
        return history

    # ---- generation / simulation ----
    def generate(
        self,
        X_new: np.ndarray,
        n_samples_per_x: int = 1,
        return_probs: bool = False,
    ) -> np.ndarray:
        """Generate Y for each row of X_new using the fitted CVAE."""
        assert self.trained, "Model must be trained before calling generate()."

        X_new = np.asarray(X_new, dtype=np.float32)
        X_new_std = self._standardize(X_new)
        n, x_dim = X_new_std.shape
        assert x_dim == self.x_dim

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_new_std).to(self.device)

            if return_probs:
                # Monte Carlo estimate of E[p(Y|X)]
                n_mc = max(n_samples_per_x, 10)
                probs_accum = torch.zeros((n, self.y_dim), device=self.device)
                for _ in range(n_mc):
                    z = torch.randn((n, self.latent_dim), device=self.device)
                    logits = self.model.decode(x_tensor, z)
                    probs = torch.sigmoid(logits)
                    probs_accum += probs
                probs_mean = probs_accum / n_mc
                return probs_mean.cpu().numpy()

            else:
                total_samples = n * n_samples_per_x
                x_rep = x_tensor.repeat_interleave(n_samples_per_x, dim=0)
                z = torch.randn((total_samples, self.latent_dim), device=self.device)
                logits = self.model.decode(x_rep, z)
                probs = torch.sigmoid(logits)
                bern = torch.distributions.Bernoulli(probs=probs)
                y_samples = bern.sample()

                y_samples_np = y_samples.cpu().numpy().astype(np.int32)

                if n_samples_per_x == 1:
                    return y_samples_np.reshape(n, self.y_dim)
                else:
                    return y_samples_np.reshape(n, n_samples_per_x, self.y_dim)


def tune_cvae_random_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    x_dim: int,
    y_dim: int,
    search_space: Dict[str, List[Any]],
    n_trials: int = 20,
    device: Optional[str] = None,
    base_seed: int = 1234,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Random-search hyperparameter tuning for CVAETrainer."""
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
            # if desired, you can add enc_hidden_dims/dec_hidden_dims to the search_space later
            hidden_dim=cfg.get("hidden_dim", 64),
            num_epochs=cfg.get("num_epochs", 50),
            batch_size=cfg.get("batch_size", 256),
            lr=cfg.get("lr", 1e-3),
            beta_kl=cfg.get("beta_kl", 1.0),
            device=device,
        )

        if verbose:
            print(f"\n=== Trial {t+1}/{n_trials} ===")
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

        trial_result = {
            "config": cfg,
            "val_loss": val_loss,
            "history": history,
        }
        trials.append(trial_result)

        if verbose:
            print(f"Trial {t+1} val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = cfg

    return {
        "trials": trials,
        "best_config": best_config,
        "best_val_loss": best_val_loss,
    }
