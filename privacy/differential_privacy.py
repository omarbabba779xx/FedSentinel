"""
Differential Privacy for FL gradient updates.
Implements: gradient clipping + Gaussian noise (DP-SGD style).
Applied on model parameter updates before sending to server.
"""

import numpy as np
import torch
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger("DifferentialPrivacy")


class DPGradientProcessor:
    """
    Applies DP to model weight updates (not per-sample gradients).
    Per-sample DP-SGD requires Opacus; this applies DP at the model update level.
    Used before sending weights to FL server.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        device: str = "cpu",
        adaptive_clipping: bool = True,
        target_unclipped_quantile: float = 0.5,
        clipping_lr: float = 0.2,
    ):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device
        self.adaptive_clipping = adaptive_clipping
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipping_lr = clipping_lr
        self._clipping_history: List[float] = []

    def clip_weights(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """L2-clip weight update."""
        flat = np.concatenate([w.flatten() for w in weights])
        norm = float(np.linalg.norm(flat))

        if norm > self.max_grad_norm:
            scale = self.max_grad_norm / norm
            clipped = [w * scale for w in weights]
        else:
            clipped = weights

        self._clipping_history.append(norm)

        if self.adaptive_clipping and len(self._clipping_history) >= 10:
            self._adapt_clipping_threshold()

        return clipped, norm

    def _adapt_clipping_threshold(self):
        """Adjust clipping threshold toward target quantile of gradient norms."""
        recent = np.array(self._clipping_history[-50:])
        quantile = float(np.quantile(recent, self.target_unclipped_quantile))
        self.max_grad_norm += self.clipping_lr * (quantile - self.max_grad_norm)
        self.max_grad_norm = max(0.01, self.max_grad_norm)

    def add_noise(self, weights: List[np.ndarray], num_clients: int = 1) -> List[np.ndarray]:
        """Add calibrated Gaussian noise to weight update."""
        noisy = []
        for w in weights:
            std = self.noise_multiplier * self.max_grad_norm / np.sqrt(max(num_clients, 1))
            noise = np.random.normal(0, std, w.shape).astype(np.float32)
            noisy.append(w + noise)
        return noisy

    def privatize(
        self,
        weights: List[np.ndarray],
        num_clients: int = 1,
    ) -> Tuple[List[np.ndarray], dict]:
        """Clip + noise in one call. Returns privatized weights + stats."""
        clipped, norm = self.clip_weights(weights)
        noisy = self.add_noise(clipped, num_clients)
        stats = {
            "gradient_norm_before": norm,
            "clipping_threshold": self.max_grad_norm,
            "noise_std": self.noise_multiplier * self.max_grad_norm / np.sqrt(max(num_clients, 1)),
            "was_clipped": norm > self.max_grad_norm,
        }
        return noisy, stats


class SecureAggregation:
    """
    Simulated Secure Aggregation (SecAgg).
    In production: cryptographic protocol (Bonawitz et al. 2017).
    Here: masks cancel out after aggregation (correctness property).
    """

    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.seed = seed

    def generate_masks(self, shapes: List[tuple]) -> List[List[np.ndarray]]:
        """
        Generate masks for each client pair such that sum of masks = 0.
        Returns list of masks per client.
        """
        rng = np.random.default_rng(self.seed)
        client_masks = [[np.zeros(s, dtype=np.float32) for s in shapes] for _ in range(self.num_clients)]

        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                pairwise_masks = [rng.standard_normal(s).astype(np.float32) for s in shapes]
                for k in range(len(shapes)):
                    client_masks[i][k] += pairwise_masks[k]
                    client_masks[j][k] -= pairwise_masks[k]

        return client_masks

    def mask_weights(
        self,
        weights: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [w + m for w, m in zip(weights, masks)]

    def aggregate_masked(self, masked_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Aggregate masked weights — masks cancel out automatically."""
        agg = [np.zeros_like(masked_updates[0][i]) for i in range(len(masked_updates[0]))]
        for client_weights in masked_updates:
            for i, w in enumerate(client_weights):
                agg[i] += w
        return [a / self.num_clients for a in agg]
