"""
Zero-Day Attack Detection via Variational Autoencoder (VAE).
Trained exclusively on NORMAL traffic — anomalies have high reconstruction error.
High reconstruction error → unknown/zero-day attack.

Combined with Isolation Forest for ensemble anomaly scoring.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("VAEDetector")


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_layer(h), self.log_var_layer(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TrafficVAE(nn.Module):
    """
    VAE for network traffic normality modelling.
    Loss = Reconstruction loss + KL divergence
    """

    def __init__(self, input_dim: int = 122, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            return F.mse_loss(x_recon, x, reduction="none").mean(dim=-1)

    @staticmethod
    def vae_loss(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss


class ZeroDayDetector:
    """
    Hybrid zero-day detector combining:
    1. VAE reconstruction error (deep anomaly detection)
    2. Isolation Forest (classical anomaly detection)
    3. Ensemble score for final decision

    Trained ONLY on normal traffic → detects anything different.
    """

    def __init__(
        self,
        input_dim: int = 122,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        device: torch.device = None,
        contamination: float = 0.05,
    ):
        self.device = device or torch.device("cpu")
        self.vae = TrafficVAE(input_dim, hidden_dim, latent_dim).to(self.device)
        self.contamination = contamination
        self.iso_forest = None
        self._vae_threshold: float = 0.0
        self._iso_threshold: float = 0.0
        self._is_fitted = False
        self.input_dim = input_dim

    def fit(
        self,
        X_normal: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta: float = 1.0,
    ) -> Dict:
        """Train VAE + Isolation Forest on normal traffic only."""
        from sklearn.ensemble import IsolationForest
        from torch.utils.data import DataLoader, TensorDataset

        logger.info(f"Training ZeroDayDetector on {len(X_normal)} normal samples")

        # Train VAE
        dataset = TensorDataset(torch.tensor(X_normal, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.vae.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_recon, mu, log_var = self.vae(batch)
                loss = TrafficVAE.vae_loss(batch, x_recon, mu, log_var, beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
            if (epoch + 1) % 10 == 0:
                logger.info(f"VAE Epoch {epoch+1}/{epochs} | loss={losses[-1]:.4f}")

        # Compute threshold from normal data (95th percentile of reconstruction error)
        recon_errors = self._vae_errors(X_normal)
        self._vae_threshold = float(np.percentile(recon_errors, 100 * (1 - self.contamination)))

        # Train Isolation Forest
        self.iso_forest = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=200)
        self.iso_forest.fit(X_normal)

        self._is_fitted = True
        logger.info(f"ZeroDayDetector fitted | VAE threshold={self._vae_threshold:.4f}")
        return {"final_loss": losses[-1], "vae_threshold": self._vae_threshold}

    def _vae_errors(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                batch = torch.tensor(X[i:i+512], dtype=torch.float32).to(self.device)
                err = self.vae.reconstruction_error(batch).cpu().numpy()
                errors.extend(err.tolist())
        return np.array(errors)

    def predict(self, X: np.ndarray) -> Dict:
        """
        Predict: -1 = anomaly (zero-day), 1 = normal
        Returns ensemble score and individual scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        vae_errors = self._vae_errors(X)
        vae_anomaly = (vae_errors > self._vae_threshold).astype(int)

        iso_scores = self.iso_forest.decision_function(X)
        iso_anomaly = (self.iso_forest.predict(X) == -1).astype(int)

        # Ensemble: anomaly if EITHER detector flags
        ensemble_anomaly = ((vae_anomaly + iso_anomaly) >= 1).astype(int)

        # Normalised anomaly score [0, 1]
        vae_norm = (vae_errors - vae_errors.min()) / (vae_errors.max() - vae_errors.min() + 1e-8)
        iso_norm = (-iso_scores - (-iso_scores).min()) / ((-iso_scores).max() - (-iso_scores).min() + 1e-8)
        ensemble_score = 0.6 * vae_norm + 0.4 * iso_norm

        return {
            "anomaly": ensemble_anomaly,
            "zero_day_score": ensemble_score,
            "vae_reconstruction_error": vae_errors,
            "isolation_forest_score": iso_scores,
            "num_anomalies": int(ensemble_anomaly.sum()),
            "anomaly_rate": float(ensemble_anomaly.mean()),
        }

    def save(self, path: str = "./results/zero_day_detector.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import pickle
        state = {
            "vae_state": self.vae.state_dict(),
            "iso_forest": self.iso_forest,
            "vae_threshold": self._vae_threshold,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"ZeroDayDetector saved to {path}")

    @classmethod
    def load(cls, path: str, input_dim: int = 122, **kwargs) -> "ZeroDayDetector":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        det = cls(input_dim=input_dim, **kwargs)
        det.vae.load_state_dict(state["vae_state"])
        det.iso_forest = state["iso_forest"]
        det._vae_threshold = state["vae_threshold"]
        det._is_fitted = state["is_fitted"]
        return det
