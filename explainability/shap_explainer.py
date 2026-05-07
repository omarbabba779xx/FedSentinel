"""
SHAP-based explainability for IDS decisions.
"Why was this connection flagged as a DoS attack?"
Uses DeepSHAP / GradientExplainer for neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("SHAPExplainer")

ATTACK_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


class FedShieldSHAPExplainer:
    """
    SHAP explainer for IDS model.
    Supports: DeepExplainer (fast, GPU), KernelExplainer (model-agnostic).
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        device: torch.device = None,
        explainer_type: str = "deep",
        n_background: int = 100,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer = None

        self.model.eval()
        self.model.to(self.device)

        bg_idx = np.random.choice(len(background_data), min(n_background, len(background_data)), replace=False)
        self.background = background_data[bg_idx]

        self._init_explainer()

    def _init_explainer(self):
        try:
            import shap
            bg_tensor = torch.tensor(self.background, dtype=torch.float32).to(self.device)

            def model_forward(x):
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
                    out = self.model(x_t)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    import torch.nn.functional as F
                    return F.softmax(logits, dim=-1).cpu().numpy()

            if self.explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, bg_tensor)
                self._use_deep = True
            else:
                self.explainer = shap.KernelExplainer(model_forward, self.background[:50])
                self._use_deep = False

            logger.info(f"SHAP {self.explainer_type} explainer initialized")

        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            self.explainer = None
            self._use_deep = False

    def explain(
        self,
        X: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute SHAP values for samples X.
        Returns shap_values array and summary dict.
        """
        if self.explainer is None:
            logger.warning("Explainer not available.")
            dummy = np.zeros_like(X)
            return dummy, {}

        import shap

        if self._use_deep:
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            shap_values = self.explainer.shap_values(x_tensor)
            if isinstance(shap_values, list):
                if class_idx is not None:
                    shap_values = np.array(shap_values[class_idx])
                else:
                    shap_values = np.array(shap_values).mean(axis=0)
        else:
            shap_values = self.explainer.shap_values(X, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).mean(axis=0)

        shap_arr = np.array(shap_values)

        mean_abs = np.abs(shap_arr).mean(axis=0) if shap_arr.ndim > 1 else np.abs(shap_arr)
        top_features_idx = np.argsort(mean_abs)[::-1][:10]

        top_features = []
        for idx in top_features_idx:
            name = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else f"feature_{idx}"
            top_features.append({"feature": name, "mean_abs_shap": float(mean_abs[idx])})

        summary = {
            "top_features": top_features,
            "mean_shap_values": mean_abs.tolist(),
            "num_samples": len(X),
        }
        return shap_arr, summary

    def explain_single(self, x: np.ndarray) -> dict:
        """Explain a single prediction with feature importance."""
        shap_vals, summary = self.explain(x.reshape(1, -1))
        shap_flat = shap_vals.flatten() if shap_vals.ndim > 1 else shap_vals

        feature_importance = {}
        for i, val in enumerate(shap_flat[:len(self.feature_names or [])]):
            name = self.feature_names[i] if self.feature_names else f"f{i}"
            feature_importance[name] = float(val)

        return {
            "feature_importance": feature_importance,
            "top_positive": {k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]},
            "top_negative": {k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1])[:5]},
        }

    def save_summary_plot(self, X: np.ndarray, output_path: str = "./results/shap_summary.png"):
        if self.explainer is None:
            return
        try:
            import shap
            import matplotlib.pyplot as plt
            shap_vals, _ = self.explain(X)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_vals, X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            logger.info(f"SHAP summary plot saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")
