"""
FedMAML: Federated Model-Agnostic Meta-Learning.
Trains a globally shared initialization θ that can quickly adapt
to a new attack type using only K gradient steps (K-shot learning).

Outer loop (server): θ ← θ - β ∇_θ Σ_i L_i(θ_i')
Inner loop (client): θ_i' = θ - α ∇_θ L_i(θ)

Reference: Finn et al. "Model-Agnostic Meta-Learning" (ICML 2017) + FL adaptation.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from utils.logger import get_logger

logger = get_logger("FedMAML")


class MAMLInnerLoop:
    """
    Inner loop optimizer for MAML (gradient through gradient).
    Uses higher-order gradients (second-order MAML).
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = True,   # FOMAML (faster, slightly less accurate)
        device: torch.device = None,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device or torch.device("cpu")

    def adapt(
        self,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        criterion: nn.Module,
    ) -> nn.Module:
        """
        Inner loop: adapt model to support set in K steps.
        Returns adapted model (does not modify original).
        """
        adapted = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)

        adapted.train()
        for _ in range(self.inner_steps):
            out = adapted(support_X.to(self.device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, support_y.to(self.device))
            optimizer.zero_grad()
            loss.backward(create_graph=not self.first_order)
            optimizer.step()

        return adapted

    def meta_loss(
        self,
        adapted_model: nn.Module,
        query_X: torch.Tensor,
        query_y: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        out = adapted_model(query_X.to(self.device))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return criterion(logits, query_y.to(self.device))


class FedMAMLServer:
    """
    MAML server: aggregates meta-gradients from clients.
    Each client sends ∇_θ L_i(θ_i') — gradient evaluated at adapted parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_lr: float = 0.001,
        device: torch.device = None,
    ):
        self.model = model
        self.outer_lr = outer_lr
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        self._round = 0
        self._history: List[dict] = []

    def aggregate_meta_gradients(
        self,
        meta_gradients: List[List[np.ndarray]],
        num_samples: List[int],
    ) -> dict:
        """
        Outer loop update: average meta-gradients across clients.
        θ ← θ - β · (1/n) Σ_i ∇_θ L_i(θ_i')
        """
        self._round += 1
        total = sum(num_samples)

        # Weighted average of meta-gradients
        avg_grads = [
            sum(meta_gradients[i][j] * num_samples[i] / total for i in range(len(meta_gradients)))
            for j in range(len(meta_gradients[0]))
        ]

        # Apply to model parameters
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), avg_grads):
            param.grad = torch.tensor(grad, dtype=torch.float32).to(self.device)
        self.optimizer.step()

        record = {"round": self._round, "num_clients": len(meta_gradients)}
        self._history.append(record)
        logger.info(f"[FedMAML Round {self._round}] Meta-update applied from {len(meta_gradients)} clients")
        return record

    def get_weights(self) -> List[np.ndarray]:
        return [p.cpu().detach().numpy() for p in self.model.parameters()]


class FedMAMLClient:
    """
    MAML client: performs inner loop adaptation + computes meta-gradient.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = True,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.inner_loop = MAMLInnerLoop(model, inner_lr, inner_steps, first_order, device)
        self.device = device or torch.device("cpu")
        self.logger = get_logger(f"FedMAML-Client-{client_id}")

    def compute_meta_gradient(
        self,
        global_weights: List[np.ndarray],
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        query_X: torch.Tensor,
        query_y: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Load global weights
        2. Inner loop: adapt to support set
        3. Compute loss on query set with adapted model
        4. Return meta-gradient (grad w.r.t. ORIGINAL params)
        """
        # Load global weights
        model = self.inner_loop.model
        keys = list(model.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in zip(keys, global_weights)})
        model.load_state_dict(state)

        # Inner loop adaptation
        adapted = self.inner_loop.adapt(support_X, support_y, criterion)

        # Query loss (meta-loss)
        query_loss = self.inner_loop.meta_loss(adapted, query_X, query_y, criterion)

        # Compute meta-gradients w.r.t. ORIGINAL model parameters
        meta_grads = torch.autograd.grad(
            query_loss,
            model.parameters(),
            allow_unused=True,
        )

        meta_grads_np = [
            g.cpu().detach().numpy() if g is not None else np.zeros_like(w)
            for g, w in zip(meta_grads, global_weights)
        ]

        num_samples = len(support_X) + len(query_X)
        metrics = {
            "client_id": self.client_id,
            "query_loss": float(query_loss.item()),
            "num_samples": num_samples,
        }
        self.logger.info(f"Meta-gradient computed | query_loss={metrics['query_loss']:.4f}")
        return meta_grads_np, num_samples, metrics

    def few_shot_predict(
        self,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        query_X: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """K-shot prediction: adapt to support set, predict on query set."""
        adapted = self.inner_loop.adapt(support_X, support_y, criterion)
        adapted.eval()
        with torch.no_grad():
            out = adapted(query_X.to(self.device))
            logits = out[0] if isinstance(out, (tuple, list)) else out
            return torch.argmax(logits, dim=-1)
