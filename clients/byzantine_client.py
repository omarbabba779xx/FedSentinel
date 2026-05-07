"""
Byzantine (malicious) client — sends poisoned model updates.
Used to simulate and test robustness of server-side defenses.
"""

import numpy as np
from typing import List, Tuple, Dict
from .base_client import FedShieldClient
from attacks import GradientPoisoningAttack, LabelFlippingAttack
from utils.logger import get_logger


class ByzantineClient(FedShieldClient):
    """
    Malicious client that:
    1. Optionally poisons its training data (label flipping)
    2. Sends poisoned gradient updates to the server
    """

    def __init__(
        self,
        client_id: int,
        attack_type: str = "sign_flip",
        label_flip_rate: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(client_id, *args, **kwargs)
        self.logger = get_logger(f"ByzantineClient-{client_id}")
        self.gradient_attack = GradientPoisoningAttack(attack_type=attack_type)
        self.label_flipper = LabelFlippingAttack(flip_rate=label_flip_rate)
        self._attack_type = attack_type
        self.logger.warning(f"[BYZANTINE] Client {client_id} is MALICIOUS | attack={attack_type}")

        # Poison training data
        self.X_train, self.y_train = self.label_flipper.poison(self.X_train, self.y_train)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        honest_params, num_samples, metrics = super().fit(parameters, config)

        # Inject poisoned gradient update
        poisoned_params = self.gradient_attack.poison(
            honest_params, honest_weights=[parameters]
        )

        metrics["is_byzantine"] = True
        metrics["attack_type"] = self._attack_type
        self.logger.warning(f"[Round {self._round}] Sent POISONED update to server")

        return poisoned_params, num_samples, metrics
