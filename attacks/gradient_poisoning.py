"""
Byzantine attack: gradient/model poisoning.
A malicious client sends crafted updates to corrupt the global model.
"""

import numpy as np
from typing import List
from utils.logger import get_logger

logger = get_logger("GradientPoisoning")


class GradientPoisoningAttack:
    """
    Gradient poisoning strategies:
    - sign_flip:     negate all gradients (label-flip equivalent in weight space)
    - scale:         amplify gradients by factor
    - random:        replace with Gaussian noise
    - min_max:       craft update to maximize deviation from honest average (Fang et al.)
    - inner_product: maximize inner product deviation (IPM attack)
    """

    def __init__(self, attack_type: str = "sign_flip", scale_factor: float = -5.0):
        self.attack_type = attack_type
        self.scale_factor = scale_factor
        logger.warning(f"[ATTACK] Gradient poisoning initialized: {attack_type}")

    def poison(
        self,
        weights: List[np.ndarray],
        honest_weights: List[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if self.attack_type == "sign_flip":
            return self._sign_flip(weights)
        elif self.attack_type == "scale":
            return self._scale(weights)
        elif self.attack_type == "random":
            return self._random(weights)
        elif self.attack_type == "min_max" and honest_weights is not None:
            return self._min_max(weights, honest_weights)
        elif self.attack_type == "inner_product" and honest_weights is not None:
            return self._inner_product(weights, honest_weights)
        else:
            return self._sign_flip(weights)

    def _sign_flip(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return [-w * abs(self.scale_factor) for w in weights]

    def _scale(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return [w * self.scale_factor for w in weights]

    def _random(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        rng = np.random.default_rng()
        return [rng.standard_normal(w.shape).astype(np.float32) * np.std(w) * 10 for w in weights]

    def _min_max(
        self,
        weights: List[np.ndarray],
        honest_weights: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Min-Max attack: find update that maximizes distance from honest average
        while staying within a plausible norm range.
        Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL" (USENIX 2020).
        """
        honest_avg = [
            np.mean([hw[i] for hw in honest_weights], axis=0) for i in range(len(weights))
        ]
        poisoned = []
        for h_avg, w in zip(honest_avg, weights):
            direction = h_avg - w
            norm = np.linalg.norm(direction.flatten())
            if norm > 0:
                unit = direction / norm
                poisoned.append(h_avg + unit * norm * abs(self.scale_factor))
            else:
                poisoned.append(w)
        return poisoned

    def _inner_product(
        self,
        weights: List[np.ndarray],
        honest_weights: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        IPM attack: negate honest average direction.
        """
        honest_avg = [
            np.mean([hw[i] for hw in honest_weights], axis=0) for i in range(len(weights))
        ]
        n = len(honest_weights) + 1
        return [-n * h for h in honest_avg]
