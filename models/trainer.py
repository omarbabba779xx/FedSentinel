"""
Training loop with cosine LR schedule, class-weighted loss, gradient clipping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from utils.logger import get_logger

logger = get_logger("Trainer")


class IDSTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        device: torch.device = None,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        self.grad_clip = grad_clip
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        self.model.to(self.device)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item() * len(y)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            out = self.model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * len(y)
            correct += (preds == y).sum().item()
            total += len(y)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> Dict:
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"lr={lr:.6f}"
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return self.history


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    lr = config.get("learning_rate", 1e-3)
    wd = config.get("weight_decay", 1e-4)
    opt_name = config.get("optimizer", "adam").lower()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, config: dict, num_epochs: int):
    sched_name = config.get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return None
