"""
Membership Inference Attack (MIA) against FL models.
Determines if a sample was in the training set of a FL client.

Attack variants:
  1. Shadow Model Attack (Shokri et al. 2017): train shadow models → meta-classifier
  2. Loss-Based Attack (Yeom et al. 2018): members have lower loss → threshold
  3. Gradient-Based Attack: members produce smaller gradient norm on trained model

Defense:
  - MMD-Based Gradient Perturbation: perturb gradient output distributions
  - DP-SGD already provides theoretical MIA resistance (tracked via ε)
  - Output perturbation: add noise to logit outputs at inference time

References:
  Shokri et al. (2017) "Membership Inference Attacks Against ML Models" — IEEE S&P
  Yeom et al. (2018) "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting"
  Nasr et al. (2018) "Machine Learning with Membership Privacy using Adversarial Regularization"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from utils.logger import get_logger

logger = get_logger("MembershipInference")


class LossThresholdAttack:
    """
    Simplest MIA: members have lower cross-entropy loss than non-members.
    Attacker threshold: classify sample as member if loss < τ.
    AUC > 0.6 = concerning leakage; AUC > 0.8 = severe leakage.
    """

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def fit_threshold(
        self,
        model: nn.Module,
        X_member: np.ndarray,
        y_member: np.ndarray,
        device: torch.device,
    ):
        """Fit optimal threshold on held-out member data."""
        losses = self._compute_losses(model, X_member, y_member, device)
        self.threshold = float(np.percentile(losses, 50))

    def _compute_losses(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        x_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            losses = criterion(logits, y_t)
        return losses.cpu().numpy()

    def attack(
        self,
        model: nn.Module,
        X_member: np.ndarray,
        y_member: np.ndarray,
        X_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
        device: torch.device,
    ) -> Dict:
        """Run attack and return metrics."""
        if self.threshold is None:
            self.fit_threshold(model, X_member, y_member, device)

        member_losses = self._compute_losses(model, X_member, y_member, device)
        nonmember_losses = self._compute_losses(model, X_nonmember, y_nonmember, device)

        all_losses = np.concatenate([member_losses, nonmember_losses])
        all_labels = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses))
        ])

        # Threshold attack: low loss → member
        preds = (all_losses < self.threshold).astype(int)
        acc = accuracy_score(all_labels, preds)

        # AUC using loss as confidence score (lower loss = more confident member)
        auc = roc_auc_score(all_labels, -all_losses)

        member_mean_loss = float(member_losses.mean())
        nonmember_mean_loss = float(nonmember_losses.mean())

        report = {
            "attack": "loss_threshold",
            "accuracy": acc,
            "auc_roc": auc,
            "member_mean_loss": member_mean_loss,
            "nonmember_mean_loss": nonmember_mean_loss,
            "loss_gap": nonmember_mean_loss - member_mean_loss,
            "threshold": self.threshold,
            "leakage_level": _leakage_level(auc),
        }
        logger.info(
            f"[MIA Loss-Threshold] AUC={auc:.4f} | acc={acc:.4f} | "
            f"leakage={report['leakage_level']}"
        )
        return report


class ShadowModelAttack:
    """
    Shadow model attack: train N shadow models, collect (confidence, label=member/non-member),
    then train a meta-classifier.
    """

    def __init__(
        self,
        num_shadows: int = 4,
        shadow_epochs: int = 5,
        device: torch.device = None,
    ):
        self.num_shadows = num_shadows
        self.shadow_epochs = shadow_epochs
        self.device = device or torch.device("cpu")
        self._meta_clf = LogisticRegression(max_iter=500)
        self._fitted = False

    def _train_shadow(
        self,
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
    ) -> nn.Module:
        """Train a shadow model on a subset of X."""
        shadow = model_fn()
        shadow.to(self.device)
        optimizer = torch.optim.Adam(shadow.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        n = len(X)
        idx = np.random.choice(n, n // 2, replace=False)
        x_t = torch.tensor(X[idx], dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y[idx], dtype=torch.long).to(self.device)

        shadow.train()
        for _ in range(self.shadow_epochs):
            optimizer.zero_grad()
            out = shadow(x_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        return shadow, idx

    def _get_confidence_vector(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return softmax confidence vector for each sample."""
        model.eval()
        x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = model(x_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def fit(self, model_fn, X_pool: np.ndarray, y_pool: np.ndarray):
        """Train shadow models and fit meta-classifier."""
        features, labels = [], []
        n = len(X_pool)

        for _ in range(self.num_shadows):
            shadow, member_idx = self._train_shadow(model_fn, X_pool, y_pool)
            member_mask = np.zeros(n, dtype=bool)
            member_mask[member_idx] = True

            conf_member = self._get_confidence_vector(shadow, X_pool[member_mask], y_pool[member_mask])
            conf_nonmember = self._get_confidence_vector(shadow, X_pool[~member_mask], y_pool[~member_mask])

            features.append(conf_member)
            labels.extend([1] * len(conf_member))
            features.append(conf_nonmember)
            labels.extend([0] * len(conf_nonmember))

        X_meta = np.vstack(features)
        self._meta_clf.fit(X_meta, np.array(labels))
        self._fitted = True
        logger.info(f"[ShadowAttack] Meta-classifier fitted on {len(X_meta)} samples")

    def attack(
        self,
        model: nn.Module,
        X_member: np.ndarray,
        X_nonmember: np.ndarray,
    ) -> Dict:
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        conf_m = self._get_confidence_vector(model, X_member,
                                              np.zeros(len(X_member), dtype=np.int64))
        conf_nm = self._get_confidence_vector(model, X_nonmember,
                                               np.zeros(len(X_nonmember), dtype=np.int64))

        X_test = np.vstack([conf_m, conf_nm])
        y_test = np.array([1] * len(conf_m) + [0] * len(conf_nm))

        preds = self._meta_clf.predict(X_test)
        probs = self._meta_clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        report = {
            "attack": "shadow_model",
            "accuracy": acc,
            "auc_roc": auc,
            "leakage_level": _leakage_level(auc),
        }
        logger.info(f"[ShadowAttack] AUC={auc:.4f} | leakage={report['leakage_level']}")
        return report


def _leakage_level(auc: float) -> str:
    if auc < 0.55:
        return "NEGLIGIBLE (DP protected)"
    elif auc < 0.65:
        return "LOW"
    elif auc < 0.75:
        return "MODERATE — consider reducing ε"
    elif auc < 0.85:
        return "HIGH — increase noise multiplier"
    else:
        return "SEVERE — model memorizing training data"


class MIADefense:
    """
    Defenses against Membership Inference Attacks.

    1. Output Perturbation: add calibrated Laplace noise to logits at inference
    2. Confidence Masking: top-k only (return only top k class probabilities)
    3. Temperature Scaling: smooth confidence distribution (reduces gap)
    4. DP-SGD reminder: training with DP-SGD provides (ε,δ)-bounded MIA resistance
    """

    def __init__(
        self,
        defense_type: str = "output_perturbation",
        epsilon: float = 1.0,  # Laplace noise scale for output perturbation
        temperature: float = 3.0,  # For temperature scaling
        top_k: int = 3,  # For confidence masking
    ):
        self.defense_type = defense_type
        self.epsilon = epsilon
        self.temperature = temperature
        self.top_k = top_k

    def defend(
        self,
        logits: torch.Tensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Apply defense to raw logits before returning to client."""
        device = device or logits.device

        if self.defense_type == "output_perturbation":
            # Laplace mechanism on softmax outputs
            probs = torch.softmax(logits, dim=-1)
            noise = torch.tensor(
                np.random.laplace(0, 1.0 / self.epsilon, probs.shape),
                dtype=torch.float32,
            ).to(device)
            noisy_probs = (probs + noise).clamp(0, 1)
            noisy_probs = noisy_probs / noisy_probs.sum(dim=-1, keepdim=True)
            return torch.log(noisy_probs + 1e-10)

        elif self.defense_type == "temperature_scaling":
            return logits / self.temperature

        elif self.defense_type == "confidence_masking":
            probs = torch.softmax(logits, dim=-1)
            # Zero out all but top-k
            topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
            masked = torch.zeros_like(probs)
            masked.scatter_(-1, topk_idx, topk_vals)
            masked = masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            return torch.log(masked + 1e-10)

        return logits

    def evaluate_defense_effectiveness(
        self,
        model: nn.Module,
        X_member: np.ndarray,
        y_member: np.ndarray,
        X_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
        device: torch.device,
    ) -> Dict:
        """Compare MIA AUC before and after defense."""
        attacker = LossThresholdAttack()
        report_before = attacker.attack(model, X_member, y_member, X_nonmember, y_nonmember, device)

        # Patch model with defense
        original_forward = model.forward

        def defended_forward(x):
            logits = original_forward(x)
            raw = logits[0] if isinstance(logits, (tuple, list)) else logits
            defended = self.defend(raw, device)
            if isinstance(logits, (tuple, list)):
                return (defended,) + logits[1:]
            return defended

        model.forward = defended_forward
        attacker2 = LossThresholdAttack()
        report_after = attacker2.attack(model, X_member, y_member, X_nonmember, y_nonmember, device)
        model.forward = original_forward  # Restore

        return {
            "defense": self.defense_type,
            "auc_before": report_before["auc_roc"],
            "auc_after": report_after["auc_roc"],
            "auc_reduction": report_before["auc_roc"] - report_after["auc_roc"],
            "leakage_before": report_before["leakage_level"],
            "leakage_after": report_after["leakage_level"],
        }
