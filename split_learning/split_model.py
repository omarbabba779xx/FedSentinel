"""
Split Learning for resource-constrained FL clients.
Model split at a cut layer:
  - Client: layers 1..cut  (lightweight, runs on edge device)
  - Server: layers cut+1..N (heavy, runs on server)

Client sends activations (smashed data) → server completes forward pass.
Server backpropagates gradients to cut layer → client updates its portion.

Privacy note: activations may leak information. Combined with DP for protection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from utils.logger import get_logger

logger = get_logger("SplitLearning")


class ClientSideModel(nn.Module):
    """
    Client-side layers (runs on edge device / client).
    Lightweight — only first few layers.
    """

    def __init__(self, input_size: int = 122, hidden_dim: int = 128, cut_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, cut_dim),
            nn.ReLU(),
        )
        self.cut_dim = cut_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ServerSideModel(nn.Module):
    """
    Server-side layers (runs on FL server).
    Receives activations from client, completes classification.
    """

    def __init__(self, cut_dim: int = 64, hidden_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cut_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.layers(activations)


class SplitLearningCoordinator:
    """
    Orchestrates split learning between client and server.
    Manages the forward/backward split protocol.
    """

    def __init__(
        self,
        client_model: ClientSideModel,
        server_model: ServerSideModel,
        device: torch.device = None,
        dp_noise_std: float = 0.0,   # add DP noise to activations
    ):
        self.client_model = client_model
        self.server_model = server_model
        self.device = device or torch.device("cpu")
        self.dp_noise_std = dp_noise_std

        self.client_model.to(self.device)
        self.server_model.to(self.device)

    def client_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Client: compute activations at cut layer."""
        x = x.to(self.device)
        activations = self.client_model(x)

        # Optional: add DP noise to activations before sending
        if self.dp_noise_std > 0 and self.training:
            noise = torch.randn_like(activations) * self.dp_noise_std
            activations = activations + noise

        return activations

    def server_forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Server: complete forward pass from cut layer to output."""
        return self.server_model(activations)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        client_optimizer: torch.optim.Optimizer,
        server_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict:
        """Complete split learning training step."""
        x, y = x.to(self.device), y.to(self.device)

        # Client forward
        self.client_model.train()
        self.server_model.train()

        client_optimizer.zero_grad()
        server_optimizer.zero_grad()

        # Step 1: client computes activations
        activations = self.client_model(x)
        activations_detached = activations.detach().requires_grad_(True)

        # Step 2: server completes forward pass
        logits = self.server_model(activations_detached)
        loss = criterion(logits, y)

        # Step 3: server backprop
        loss.backward()
        server_optimizer.step()

        # Step 4: send gradients back to client cut layer
        grad_cut = activations_detached.grad.clone()
        activations.backward(grad_cut)
        client_optimizer.step()

        preds = torch.argmax(logits.detach(), dim=-1)
        accuracy = (preds == y).float().mean().item()

        return {
            "loss": float(loss.item()),
            "accuracy": accuracy,
            "activation_norm": float(activations.detach().norm().item()),
        }

    @property
    def training(self) -> bool:
        return self.client_model.training

    def get_client_weights(self) -> List[np.ndarray]:
        return [p.cpu().detach().numpy() for p in self.client_model.parameters()]

    def get_server_weights(self) -> List[np.ndarray]:
        return [p.cpu().detach().numpy() for p in self.server_model.parameters()]

    def set_client_weights(self, weights: List[np.ndarray]):
        from collections import OrderedDict
        keys = list(self.client_model.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in zip(keys, weights)})
        self.client_model.load_state_dict(state)
