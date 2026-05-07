"""
Adversarial Robustness Training for FL (AT-FL).
Uses PGD (Projected Gradient Descent) to generate adversarial examples
during local training, making the model robust to evasion attacks.

PGD attack: x_adv = Π_{x+S} [x_adv + α·sign(∇_x L(θ, x_adv, y))]

Min-max training objective:
  min_θ E_{(x,y)} [ max_{δ:‖δ‖∞≤ε} L(θ, x+δ, y) ]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("AdversarialTraining")


class PGDAttack:
    """
    PGD adversarial attack for generating adversarial examples.
    Supports L∞ and L2 norms.
    """

    def __init__(
        self,
        eps: float = 0.1,            # perturbation budget
        alpha: float = 0.01,          # step size
        num_steps: int = 10,          # PGD iterations
        norm: str = "linf",           # linf | l2
        random_start: bool = True,
    ):
        self.eps = eps
        self.alpha = alpha
        self.num_steps = num_steps
        self.norm = norm
        self.random_start = random_start

    def perturb(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        x = x.to(device)
        y = y.to(device)

        if self.random_start:
            if self.norm == "linf":
                delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            else:
                delta = torch.randn_like(x) * self.eps
            x_adv = (x + delta).clamp(x.min().item(), x.max().item())
        else:
            x_adv = x.clone()

        x_adv = x_adv.detach().requires_grad_(True)

        for _ in range(self.num_steps):
            out = model(x_adv)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            if self.norm == "linf":
                x_adv = (x_adv + self.alpha * grad.sign()).detach()
                delta = (x_adv - x).clamp(-self.eps, self.eps)
                x_adv = (x + delta).clamp(x.min().item(), x.max().item())
            else:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                x_adv = (x_adv + self.alpha * grad / grad_norm).detach()
                delta = x_adv - x
                delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                delta = delta * torch.clamp(self.eps / delta_norm, max=1.0)
                x_adv = (x + delta).clamp(x.min().item(), x.max().item())

            x_adv = x_adv.detach().requires_grad_(True)

        return x_adv.detach()


class AdversarialTrainer:
    """
    Adversarial training wrapper for FL clients.
    Each training batch mixes clean + adversarial examples.

    AT loss = (1-β)·L_clean + β·L_adv
    """

    def __init__(
        self,
        pgd: PGDAttack,
        adv_ratio: float = 0.5,      # fraction of batch to make adversarial
        device: torch.device = None,
    ):
        self.pgd = pgd
        self.adv_ratio = adv_ratio
        self.device = device or torch.device("cpu")

    def adversarial_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        x, y = x.to(self.device), y.to(self.device)

        # Split batch into clean + adversarial
        n_adv = int(len(x) * self.adv_ratio)
        x_clean, y_clean = x[n_adv:], y[n_adv:]
        x_adv_input, y_adv = x[:n_adv], y[:n_adv]

        # Generate adversarial examples
        model.eval()
        x_adv = self.pgd.perturb(model, x_adv_input, y_adv, criterion, self.device)
        model.train()

        # Compute losses
        out_clean = model(x_clean)
        logits_clean = out_clean[0] if isinstance(out_clean, (tuple, list)) else out_clean
        loss_clean = criterion(logits_clean, y_clean) if len(x_clean) > 0 else torch.tensor(0.0)

        out_adv = model(x_adv)
        logits_adv = out_adv[0] if isinstance(out_adv, (tuple, list)) else out_adv
        loss_adv = criterion(logits_adv, y_adv) if n_adv > 0 else torch.tensor(0.0)

        total_loss = (1 - self.adv_ratio) * loss_clean + self.adv_ratio * loss_adv

        stats = {
            "loss_clean": float(loss_clean.item()),
            "loss_adv": float(loss_adv.item()),
            "total_loss": float(total_loss.item()),
            "n_adv_examples": n_adv,
        }
        return total_loss, stats

    def train_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        grad_clip: float = 1.0,
    ) -> dict:
        model.train()
        total_stats = {"loss_clean": 0, "loss_adv": 0, "total_loss": 0, "n_batches": 0}

        for x, y in loader:
            optimizer.zero_grad()
            loss, stats = self.adversarial_loss(model, x, y, criterion)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for k in ["loss_clean", "loss_adv", "total_loss"]:
                total_stats[k] += stats[k]
            total_stats["n_batches"] += 1

        n = max(total_stats["n_batches"], 1)
        return {k: v / n for k, v in total_stats.items() if k != "n_batches"}
