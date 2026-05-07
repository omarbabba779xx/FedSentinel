"""
Label flipping attack: corrupt training data by flipping labels.
Source labels (e.g. 'normal') → target labels (e.g. 'DoS').
"""

import numpy as np
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("LabelFlipping")


class LabelFlippingAttack:
    """
    Label flipping strategies:
    - targeted:      flip specific source class → target class
    - random:        randomly flip a fraction of labels
    - backdoor:      flip all source-class labels to target
    """

    def __init__(
        self,
        attack_type: str = "targeted",
        source_class: int = 0,      # 0 = Normal
        target_class: int = 1,      # 1 = DoS
        flip_rate: float = 1.0,     # fraction of source samples to flip
        seed: int = 42,
    ):
        self.attack_type = attack_type
        self.source_class = source_class
        self.target_class = target_class
        self.flip_rate = flip_rate
        self.rng = np.random.default_rng(seed)
        logger.warning(f"[ATTACK] Label flipping: {attack_type} | {source_class}→{target_class} | rate={flip_rate}")

    def poison(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_poisoned = y.copy()

        if self.attack_type == "targeted":
            source_idx = np.where(y == self.source_class)[0]
            n_flip = int(len(source_idx) * self.flip_rate)
            flip_idx = self.rng.choice(source_idx, n_flip, replace=False)
            y_poisoned[flip_idx] = self.target_class
            logger.info(f"Flipped {n_flip} labels: {self.source_class} → {self.target_class}")

        elif self.attack_type == "random":
            n_flip = int(len(y) * self.flip_rate)
            flip_idx = self.rng.choice(len(y), n_flip, replace=False)
            num_classes = int(y.max()) + 1
            random_labels = self.rng.integers(0, num_classes, n_flip)
            y_poisoned[flip_idx] = random_labels
            logger.info(f"Random label flip: {n_flip} samples")

        elif self.attack_type == "backdoor":
            y_poisoned[y == self.source_class] = self.target_class
            count = int((y == self.source_class).sum())
            logger.info(f"Backdoor flip: all {count} class-{self.source_class} → class-{self.target_class}")

        return X, y_poisoned

    def poison_with_trigger(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trigger_features: Optional[np.ndarray] = None,
        trigger_fraction: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backdoor with feature trigger: inject a pattern into a fraction of samples.
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        n_poison = int(len(X) * trigger_fraction)
        poison_idx = self.rng.choice(len(X), n_poison, replace=False)

        if trigger_features is None:
            trigger_features = np.ones(X.shape[1], dtype=np.float32) * 999.0

        X_poisoned[poison_idx] = trigger_features
        y_poisoned[poison_idx] = self.target_class

        logger.info(f"Backdoor trigger injected into {n_poison} samples")
        return X_poisoned, y_poisoned
