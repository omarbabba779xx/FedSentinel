"""
Free-rider attack: client contributes no real computation.
Sends last round's global model or Gaussian noise.
"""

import numpy as np
from typing import List, Optional
from utils.logger import get_logger

logger = get_logger("FreeRider")


class FreeRiderAttack:
    """
    Free-rider strategies:
    - delta_weights: send near-zero delta (pretend no local training happened)
    - random_noise:  send random Gaussian noise
    - replay:        resend previous global weights unchanged
    - disguise:      add small noise to global weights to avoid detection
    """

    def __init__(self, strategy: str = "delta_weights", noise_std: float = 1e-4, seed: int = 42):
        self.strategy = strategy
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self._previous_weights: Optional[List[np.ndarray]] = None
        logger.warning(f"[ATTACK] Free-rider initialized: {strategy}")

    def fake_update(
        self,
        global_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        if self.strategy == "delta_weights":
            return [w + self.rng.normal(0, self.noise_std, w.shape).astype(np.float32) for w in global_weights]

        elif self.strategy == "random_noise":
            return [self.rng.standard_normal(w.shape).astype(np.float32) * self.noise_std for w in global_weights]

        elif self.strategy == "replay":
            if self._previous_weights is not None:
                return self._previous_weights
            return global_weights

        elif self.strategy == "disguise":
            scale = self.rng.uniform(0.99, 1.01)
            noise = [self.rng.normal(0, self.noise_std, w.shape).astype(np.float32) for w in global_weights]
            result = [w * scale + n for w, n in zip(global_weights, noise)]
            self._previous_weights = result
            return result

        return global_weights
