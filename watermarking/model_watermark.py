"""
Neural Network Model Watermarking.
Embeds an invisible signature into the model during FL training.
Proves model ownership if the model is stolen/leaked.

Method: Backdoor-based watermarking
  - Watermark key = set of (x_wm, y_wm) trigger samples
  - Model trained to classify trigger patterns → specific target label
  - Ownership verified by: model(x_wm) == y_wm with prob > threshold
  - Triggers are unique per FL session → unforgeable

Reference: Adi et al. "Turning Your Weakness Into a Strength: Watermarking DNN..." (USENIX Security 2018)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import hashlib
import json
from utils.logger import get_logger

logger = get_logger("ModelWatermark")


class WatermarkKey:
    """Watermark key: set of trigger (input, label) pairs."""

    def __init__(
        self,
        trigger_inputs: np.ndarray,
        trigger_labels: np.ndarray,
        owner_id: str,
        session_id: str,
    ):
        self.trigger_inputs = trigger_inputs
        self.trigger_labels = trigger_labels
        self.owner_id = owner_id
        self.session_id = session_id
        self.key_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        combined = self.trigger_inputs.tobytes() + self.trigger_labels.tobytes()
        return hashlib.sha256(combined).hexdigest()

    def save(self, path: str):
        np.savez(
            path,
            trigger_inputs=self.trigger_inputs,
            trigger_labels=self.trigger_labels,
        )
        meta = {"owner_id": self.owner_id, "session_id": self.session_id, "key_hash": self.key_hash}
        with open(path + "_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "WatermarkKey":
        data = np.load(path + ".npz")
        with open(path + "_meta.json") as f:
            meta = json.load(f)
        return cls(data["trigger_inputs"], data["trigger_labels"], meta["owner_id"], meta["session_id"])


class WatermarkGenerator:
    """Generates diverse watermark patterns."""

    @staticmethod
    def random_feature_pattern(
        num_triggers: int = 50,
        input_dim: int = 122,
        target_label: int = 0,
        seed: int = 42,
        owner_id: str = "FedSentinel",
        session_id: str = "round_0",
    ) -> WatermarkKey:
        """Random feature patterns as triggers (independent of training data)."""
        rng = np.random.default_rng(seed)
        triggers = rng.standard_normal((num_triggers, input_dim)).astype(np.float32)
        triggers *= 10.0  # Out-of-distribution magnitude
        labels = np.full(num_triggers, target_label, dtype=np.int64)
        return WatermarkKey(triggers, labels, owner_id, session_id)

    @staticmethod
    def content_pattern(
        X_normal: np.ndarray,
        num_triggers: int = 50,
        target_label: int = 0,
        trigger_features: np.ndarray = None,
        seed: int = 42,
        owner_id: str = "FedSentinel",
        session_id: str = "round_0",
    ) -> WatermarkKey:
        """Embed watermark pattern into real samples."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_normal), num_triggers, replace=False)
        triggers = X_normal[idx].copy()

        if trigger_features is None:
            trigger_features = rng.standard_normal(X_normal.shape[1]).astype(np.float32) * 5.0

        triggers[:, -5:] = trigger_features[-5:]  # Embed in last 5 features
        labels = np.full(num_triggers, target_label, dtype=np.int64)
        return WatermarkKey(triggers, labels, owner_id, session_id)


class WatermarkEmbedder:
    """Embeds watermark into model during FL training."""

    def __init__(
        self,
        key: WatermarkKey,
        embed_frequency: int = 5,   # embed every N rounds
        embed_epochs: int = 3,
        lr: float = 1e-4,
        device: torch.device = None,
    ):
        self.key = key
        self.embed_frequency = embed_frequency
        self.embed_epochs = embed_epochs
        self.lr = lr
        self.device = device or torch.device("cpu")

    def embed(
        self,
        model: nn.Module,
        criterion: nn.Module,
        round_num: int,
    ) -> Dict:
        """Embed watermark by fine-tuning on trigger samples."""
        if round_num % self.embed_frequency != 0:
            return {"embedded": False}

        x_wm = torch.tensor(self.key.trigger_inputs, dtype=torch.float32).to(self.device)
        y_wm = torch.tensor(self.key.trigger_labels, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()

        for _ in range(self.embed_epochs):
            optimizer.zero_grad()
            out = model(x_wm)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, y_wm)
            loss.backward()
            optimizer.step()

        acc = self.verify(model)
        logger.info(f"Watermark embedded at round {round_num} | trigger_accuracy={acc:.3f}")
        return {"embedded": True, "round": round_num, "trigger_accuracy": acc}

    @torch.no_grad()
    def verify(self, model: nn.Module, threshold: float = 0.8) -> float:
        """Verify watermark presence. Returns trigger accuracy."""
        model.eval()
        x_wm = torch.tensor(self.key.trigger_inputs, dtype=torch.float32).to(self.device)
        y_wm = torch.tensor(self.key.trigger_labels, dtype=torch.long).to(self.device)

        out = model(x_wm)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == y_wm).float().mean().item()
        return accuracy


class OwnershipVerifier:
    """Verifies model ownership using watermark key."""

    def __init__(self, embedder: WatermarkEmbedder, threshold: float = 0.8):
        self.embedder = embedder
        self.threshold = threshold

    def verify_ownership(self, model: nn.Module) -> Dict:
        """Run ownership verification and return report."""
        accuracy = self.embedder.verify(model)
        is_owner = accuracy >= self.threshold

        report = {
            "is_owner": is_owner,
            "trigger_accuracy": accuracy,
            "threshold": self.threshold,
            "key_hash": self.embedder.key.key_hash,
            "owner_id": self.embedder.key.owner_id,
            "session_id": self.embedder.key.session_id,
            "verdict": "OWNERSHIP CONFIRMED" if is_owner else "OWNERSHIP NOT CONFIRMED",
        }
        logger.info(f"Ownership verification: {report['verdict']} (accuracy={accuracy:.3f})")
        return report
