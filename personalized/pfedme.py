"""
pFedMe: Personalized Federated Learning with Moreau Envelopes.
Decouples personalized model optimization from global model update.

Objective: min_w (1/n) Σ F_i*(w)
where F_i*(w) = min_{θ_i} [f_i(θ_i) + (λ/2)‖θ_i - w‖²]

Reference: Dinh et al. "Personalized Federated Learning with Moreau Envelopes" (NeurIPS 2020)
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict
from utils.logger import get_logger

logger = get_logger("pFedMe")


class pFedMeClient:
    """
    pFedMe client: solves proximal problem via K steps of SGD per round.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        lam: float = 15.0,          # regularization (λ) — higher = closer to global
        local_lr: float = 0.01,
        K: int = 5,                  # inner loop steps per round
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.lam = lam
        self.local_lr = local_lr
        self.K = K
        self.device = device or torch.device("cpu")

        self.personal_model = model.to(self.device)
        self.logger = get_logger(f"pFedMe-{client_id}")

    def local_update(
        self,
        global_weights: List[np.ndarray],
        train_loader,
        criterion: nn.Module,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Solve proximal problem for K steps:
        θ_i^* = argmin f_i(θ) + (λ/2)‖θ - w‖²
        Return: gradient approximation for global model update.
        """
        keys = list(self.personal_model.state_dict().keys())
        w_global = [torch.tensor(gw, device=self.device) for gw in global_weights]

        # Initialize personal model from global
        state = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in zip(keys, global_weights)})
        self.personal_model.load_state_dict(state)

        optimizer = torch.optim.SGD(self.personal_model.parameters(), lr=self.local_lr)
        total_loss = 0.0

        # K inner steps
        data_iter = iter(train_loader)
        for _ in range(self.K):
            try:
                X, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                X, y = next(data_iter)

            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            out = self.personal_model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            task_loss = criterion(logits, y)

            # Proximal term
            prox = sum(
                torch.norm(p - gw) ** 2
                for p, gw in zip(self.personal_model.parameters(), w_global)
            )
            loss = task_loss + (self.lam / 2.0) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Gradient approximation = λ(w - θ*)
        personal_params = list(self.personal_model.parameters())
        grad_approx = [
            (self.lam * (gw - p.detach())).cpu().numpy()
            for gw, p in zip(w_global, personal_params)
        ]

        # Updated global weights = w - lr * grad_approx
        updated_global = [
            gw.cpu().numpy() - self.local_lr * g
            for gw, g in zip(w_global, grad_approx)
        ]

        metrics = {
            "client_id": self.client_id,
            "avg_loss": total_loss / self.K,
            "lambda": self.lam,
        }
        return updated_global, metrics

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict:
        self.personal_model.eval()
        correct, total = 0, 0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            out = self.personal_model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
        return {"accuracy": correct / max(total, 1)}
