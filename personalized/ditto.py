"""
Ditto: Personalized Federated Learning.
Each client maintains a personal model v_i alongside the global model w.

Objective per client i:
  min_{v_i} h_i(v_i) + (λ/2) ‖v_i - w‖²

The personal model v_i fine-tunes toward client's local distribution
while staying close to the global model via the proximal term λ.

Reference: Li et al. "Ditto: Fair and Robust Federated Learning Through Personalization" (ICML 2021)
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from utils.logger import get_logger

logger = get_logger("Ditto")


class DittoClient:
    """
    Ditto personalized FL client.
    Maintains both global model w and personal model v_i.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        lam: float = 0.1,           # proximal term strength (λ)
        personal_lr: float = 0.01,
        global_epochs: int = 3,
        personal_epochs: int = 5,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.lam = lam
        self.personal_lr = personal_lr
        self.global_epochs = global_epochs
        self.personal_epochs = personal_epochs
        self.device = device or torch.device("cpu")
        self.logger = get_logger(f"Ditto-Client-{client_id}")

        self.global_model = model.to(self.device)
        self.personal_model = copy.deepcopy(model).to(self.device)

    def update_global(
        self,
        global_weights: List[np.ndarray],
        train_loader,
        criterion: nn.Module,
    ) -> List[np.ndarray]:
        """Standard FL update: train on local data, return new weights."""
        keys = list(self.global_model.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in zip(keys, global_weights)})
        self.global_model.load_state_dict(state)

        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=1e-3)
        self.global_model.train()

        for _ in range(self.global_epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.global_model(X)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        return [p.cpu().detach().numpy() for p in self.global_model.parameters()]

    def update_personal(
        self,
        global_weights: List[np.ndarray],
        train_loader,
        criterion: nn.Module,
    ) -> Dict:
        """
        Personal model update with proximal term:
        L_personal = L_task(v_i) + (λ/2) ‖v_i - w‖²
        """
        # Load global weights as reference (w)
        global_params = [torch.tensor(gw).to(self.device) for gw in global_weights]

        optimizer = torch.optim.SGD(self.personal_model.parameters(), lr=self.personal_lr)
        self.personal_model.train()

        total_loss = 0.0
        n_batches = 0

        for _ in range(self.personal_epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                out = self.personal_model(X)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                task_loss = criterion(logits, y)

                # Proximal term: (λ/2) Σ ‖v_i - w‖²
                proximal = sum(
                    torch.norm(p - gp) ** 2
                    for p, gp in zip(self.personal_model.parameters(), global_params)
                )
                loss = task_loss + (self.lam / 2.0) * proximal

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.personal_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.logger.info(f"Personal model updated | avg_loss={avg_loss:.4f} | λ={self.lam}")
        return {"personal_loss": avg_loss, "client_id": self.client_id}

    @torch.no_grad()
    def evaluate_personal(self, val_loader) -> Dict:
        self.personal_model.eval()
        correct, total = 0, 0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            out = self.personal_model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
        return {"personal_accuracy": correct / max(total, 1), "client_id": self.client_id}

    def predict(self, x: torch.Tensor, use_personal: bool = True) -> torch.Tensor:
        model = self.personal_model if use_personal else self.global_model
        model.eval()
        with torch.no_grad():
            out = model(x.to(self.device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            return torch.argmax(logits, dim=-1)

    def save_personal_model(self, path: str):
        torch.save(self.personal_model.state_dict(), path)

    def load_personal_model(self, path: str):
        self.personal_model.load_state_dict(torch.load(path, map_location=self.device))
