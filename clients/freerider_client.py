"""
Free-rider client — does no real training, sends fake updates.
"""

import numpy as np
from typing import List, Tuple
from .base_client import FedShieldClient
from attacks import FreeRiderAttack
from utils.logger import get_logger


class FreeRiderClient(FedShieldClient):
    def __init__(
        self,
        client_id: int,
        strategy: str = "delta_weights",
        *args,
        **kwargs,
    ):
        super().__init__(client_id, *args, **kwargs)
        self.logger = get_logger(f"FreeRiderClient-{client_id}")
        self.attack = FreeRiderAttack(strategy=strategy)
        self._strategy = strategy
        self.logger.warning(f"[FREE-RIDER] Client {client_id} strategy={strategy}")

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        fake_params = self.attack.fake_update(parameters)
        self._round += 1

        metrics = {
            "client_id": self.client_id,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "epsilon": 0.0,
            "round": self._round,
            "num_samples": len(self.X_train),
            "is_free_rider": True,
            "strategy": self._strategy,
        }
        return fake_params, len(self.X_train), metrics
