"""
Split Learning integrated with Flower FL framework.
SplitFedClient: Flower NumPyClient that only holds the client-side model
                (up to the cut layer). Sends smashed data to server.

Architecture:
  Client side: input → cut_layer   → smashed_data (sent to server)
  Server side: smashed_data → rest → loss → grad_smash (sent back)

This is "SplitFed" (Thapa et al. 2022): combines privacy of split learning
with scalability of federated learning.

Reference: Thapa et al. (2022) "SplitFed: When Federated Learning Meets Split Learning"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import get_logger

logger = get_logger("SplitFedClient")


class SplitFedClientModel(nn.Module):
    """Client-side model: first N layers only."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, cut_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cut_dim),
            nn.ReLU(),
        )
        self.cut_dim = cut_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SplitFedServerModel(nn.Module):
    """Server-side model: from cut layer to output."""

    def __init__(self, cut_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cut_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        return self.layers(smashed)


class SplitFedFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementing SplitFed protocol.
    The client:
      1. Receives server model params (full round sync via FedAvg on client-side models)
      2. Runs local split-learning steps (forward → send smash → receive grad → backward)
      3. Returns updated client-side model params
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        client_model: SplitFedClientModel,
        server_model_local: SplitFedServerModel,  # Local copy of server model
        num_classes: int = 5,
        local_epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.client_model = client_model
        self.server_model_local = server_model_local
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        self.client_model.to(self.device)
        self.server_model_local.to(self.device)

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=batch_size, shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=batch_size, shuffle=False,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.client_optim = torch.optim.Adam(self.client_model.parameters(), lr=lr)
        self.server_optim = torch.optim.Adam(self.server_model_local.parameters(), lr=lr)

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        """Return client-side model parameters only (smashed data not sent)."""
        return [p.detach().cpu().numpy() for p in self.client_model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load global client-side model parameters."""
        for param, new_val in zip(self.client_model.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=torch.float32).to(self.device)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        """Local SplitFed training round."""
        self.set_parameters(parameters)

        self.client_model.train()
        self.server_model_local.train()

        total_loss = 0.0
        n_batches = 0

        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Client-side forward
                self.client_optim.zero_grad()
                self.server_optim.zero_grad()

                smashed = self.client_model(X_batch)
                smashed_detached = smashed.detach().requires_grad_(True)

                # Server-side forward (simulated locally)
                server_out = self.server_model_local(smashed_detached)
                loss = self.criterion(server_out, y_batch)
                loss.backward()

                # Pass gradient back to client
                smashed.backward(smashed_detached.grad)

                self.client_optim.step()
                self.server_optim.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"[SplitFed Client {self.client_id}] loss={avg_loss:.4f}")

        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        """Local evaluation on validation set."""
        self.set_parameters(parameters)
        self.client_model.eval()
        self.server_model_local.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                smashed = self.client_model(X_batch)
                out = self.server_model_local(smashed)
                loss = self.criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = torch.argmax(out, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return avg_loss, total, {"accuracy": accuracy}


def build_splitfed_client(
    client_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int = 5,
    cut_dim: int = 64,
    device: torch.device = None,
) -> SplitFedFlowerClient:
    """Factory: build SplitFedFlowerClient with default architecture."""
    client_model = SplitFedClientModel(input_dim=input_dim, cut_dim=cut_dim)
    server_model_local = SplitFedServerModel(cut_dim=cut_dim, num_classes=num_classes)
    return SplitFedFlowerClient(
        client_id=client_id,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        client_model=client_model,
        server_model_local=server_model_local,
        num_classes=num_classes,
        device=device,
    )
