"""
Base Flower FL client. Wraps model training and parameter exchange.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import flwr as fl

from models import build_model, IDSTrainer, build_optimizer, build_scheduler
from data.dataset import IDSDataset, make_dataloaders, train_val_split, compute_class_weights
from privacy import DPGradientProcessor, PrivacyAccountant
from utils.logger import get_logger
from utils.helpers import get_device


class FedShieldClient(fl.client.NumPyClient):
    """
    Base FL client for FedShield-IDS.
    Handles:
    - Local training with full model pipeline
    - DP noise injection on weight updates
    - Privacy budget tracking
    - Metrics reporting to server
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_config: dict,
        client_config: dict,
        num_classes: int = 5,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_config = model_config
        self.client_config = client_config
        self.num_classes = num_classes
        self.device = get_device()
        self.logger = get_logger(f"Client-{client_id}")

        input_size = X_train.shape[1]
        self.model = build_model(
            architecture=client_config.get("model", {}).get("architecture", "transformer"),
            input_size=input_size,
            num_classes=num_classes,
        ).to(self.device)

        dp_cfg = client_config.get("privacy", {})
        self.dp_enabled = dp_cfg.get("enabled", True)
        self.dp_processor = DPGradientProcessor(
            max_grad_norm=dp_cfg.get("max_grad_norm", 1.0),
            noise_multiplier=dp_cfg.get("noise_multiplier", 1.1),
            adaptive_clipping=dp_cfg.get("adaptive_clipping", True),
        ) if self.dp_enabled else None

        self.privacy_accountant = PrivacyAccountant(
            target_epsilon=dp_cfg.get("epsilon", 1.0),
            target_delta=dp_cfg.get("delta", 1e-5),
        ) if self.dp_enabled else None

        self.train_loader, self.val_loader = make_dataloaders(
            X_train, y_train, X_val, y_val,
            batch_size=client_config.get("batch_size", 256),
        )

        class_weights = compute_class_weights(y_train, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )

        self._round = 0

    def get_parameters(self, config: dict = None) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self._round += 1
        self.set_parameters(parameters)

        optimizer = build_optimizer(self.model, self.client_config)
        scheduler = build_scheduler(optimizer, self.client_config, epochs=self.client_config.get("local_epochs", 3))

        trainer = IDSTrainer(
            model=self.model,
            optimizer=optimizer,
            criterion=self.criterion,
            scheduler=scheduler,
            device=self.device,
            grad_clip=self.client_config.get("gradient_clip", 1.0),
        )

        local_epochs = int(config.get("local_epochs", self.client_config.get("local_epochs", 3)))
        history = trainer.fit(self.train_loader, self.val_loader, epochs=local_epochs, verbose=False)

        updated_params = self.get_parameters()

        if self.dp_enabled and self.dp_processor is not None:
            weight_delta = [
                up - bp for up, bp in zip(updated_params, parameters)
            ]
            privatized_delta, dp_stats = self.dp_processor.privatize(
                weight_delta, num_clients=1
            )
            updated_params = [
                bp + pd for bp, pd in zip(parameters, privatized_delta)
            ]

            sample_rate = len(self.X_train) / (len(self.X_train) * local_epochs)
            eps, _ = self.privacy_accountant.step(
                self.dp_processor.noise_multiplier,
                sample_rate=min(1.0, len(self.train_loader.dataset) / len(self.X_train)),
                num_steps=local_epochs * len(self.train_loader),
            )
        else:
            eps = 0.0
            dp_stats = {}

        val_metrics = trainer.evaluate(self.val_loader)
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(history["train_loss"][-1]) if history["train_loss"] else 0.0,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "epsilon": float(eps),
            "round": self._round,
            "num_samples": len(self.X_train),
        }

        self.logger.info(
            f"[Round {self._round}] fit complete | "
            f"val_acc={val_metrics['accuracy']:.4f} | ε={eps:.4f}"
        )
        return updated_params, len(self.X_train), metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = self.criterion(logits, y)
                preds = torch.argmax(logits, dim=-1)
                total_loss += loss.item() * len(y)
                correct += (preds == y).sum().item()
                total += len(y)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return avg_loss, total, {"accuracy": accuracy, "client_id": self.client_id}
