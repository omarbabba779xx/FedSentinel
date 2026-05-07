"""
LIME-based explainability for individual IDS predictions.
Model-agnostic, works with any classifier.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from utils.logger import get_logger

logger = get_logger("LIMEExplainer")

ATTACK_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


class FedShieldLIMEExplainer:
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        device: torch.device = None,
        num_features: int = 10,
        num_samples: int = 1000,
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ATTACK_NAMES
        self.device = device or torch.device("cpu")
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = None

        self.model.eval()
        self.model.to(self.device)
        self._init()

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(x_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            import torch.nn.functional as F
            return F.softmax(logits, dim=-1).cpu().numpy()

    def _init(self):
        try:
            from lime.lime_tabular import LimeTabularExplainer
            # Need training data for LIME; use zeros as placeholder
            self._LimeTabularExplainer = LimeTabularExplainer
            logger.info("LIME explainer ready")
        except ImportError:
            logger.warning("LIME not installed. Run: pip install lime")
            self._LimeTabularExplainer = None

    def explain(
        self,
        x: np.ndarray,
        training_data: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> dict:
        if self._LimeTabularExplainer is None:
            return {"error": "LIME not installed"}

        explainer = self._LimeTabularExplainer(
            training_data=training_data,
            mode="classification",
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=True,
        )

        exp = explainer.explain_instance(
            x.flatten(),
            self._predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            top_labels=len(self.class_names),
        )

        label = class_idx if class_idx is not None else exp.top_labels[0]
        feature_weights = dict(exp.as_list(label=label))

        proba = self._predict_fn(x.reshape(1, -1))[0]
        predicted_class = int(np.argmax(proba))

        return {
            "predicted_class": predicted_class,
            "predicted_class_name": self.class_names[predicted_class] if predicted_class < len(self.class_names) else str(predicted_class),
            "probabilities": {self.class_names[i]: float(proba[i]) for i in range(len(proba))},
            "feature_weights": feature_weights,
            "explanation_label": self.class_names[label] if label < len(self.class_names) else str(label),
        }
