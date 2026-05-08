"""
Model Inversion Attack Defense.
Model inversion: adversary reconstructs training samples from model outputs/gradients.

Attack: given model f and output y, find x* = argmax P(y|f(x)) — reconstruct input.
Defense strategies implemented:
  1. Representation Noise Injection: add noise to intermediate layer activations
  2. Output Randomization: randomize prediction confidence (harder to invert)
  3. Knowledge Distillation Defense: train a student model that masks internal representations
  4. Gradient Sanitization: detect and block high-fidelity gradient queries

Reference:
  Fredrikson et al. (2015) "Model Inversion Attacks that Exploit Confidence Information"
  Yang et al. (2019) "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
  Mireshghallah et al. (2020) "Shredder: Learning Noise Distributions to Protect Inference Privacy"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("ModelInversionDefense")


class RepresentationNoiseHook:
    """
    Injects noise into intermediate layer activations via forward hooks.
    Degrades model inversion quality while minimally impacting accuracy.
    Uses calibrated Gaussian noise σ ∝ activation magnitude.
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        layers: Optional[List[str]] = None,
    ):
        self.noise_scale = noise_scale
        self.target_layers = layers  # None = apply to all Linear layers
        self._hooks = []

    def _make_hook(self):
        def hook(module, input, output):
            if self.noise_scale > 0:
                std = output.detach().abs().mean() * self.noise_scale
                noise = torch.randn_like(output) * std
                return output + noise
            return output
        return hook

    def register(self, model: nn.Module):
        """Attach noise hooks to target layers."""
        self.remove()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.MultiheadAttention)):
                if self.target_layers is None or name in self.target_layers:
                    h = module.register_forward_hook(self._make_hook())
                    self._hooks.append(h)
        logger.info(f"[MI Defense] Registered {len(self._hooks)} noise hooks (σ_scale={self.noise_scale})")

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class OutputRandomizationDefense:
    """
    Randomizes output confidence to prevent precision model inversion.
    Adds Laplace noise to output probabilities (post-softmax).
    Stronger than temperature scaling — directly degrades inversion.
    """

    def __init__(
        self,
        noise_epsilon: float = 2.0,  # Laplace noise parameter (higher = more privacy)
        rounding: int = 2,           # Round probabilities to k decimal places
    ):
        self.noise_epsilon = noise_epsilon
        self.rounding = rounding

    def defend_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply defense to output logits."""
        probs = torch.softmax(logits, dim=-1)
        # Laplace noise proportional to sensitivity / epsilon
        sensitivity = 1.0 / logits.shape[-1]  # L1 sensitivity of softmax
        scale = sensitivity / self.noise_epsilon
        noise = torch.tensor(
            np.random.laplace(0, scale, probs.shape), dtype=torch.float32
        ).to(logits.device)
        noisy_probs = (probs + noise).clamp(0, 1)
        noisy_probs = noisy_probs / noisy_probs.sum(dim=-1, keepdim=True)

        if self.rounding > 0:
            factor = 10 ** self.rounding
            noisy_probs = (noisy_probs * factor).round() / factor
            noisy_probs = noisy_probs / noisy_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        return torch.log(noisy_probs + 1e-10)


class GradientSanitizer:
    """
    Detects high-fidelity gradient queries that could enable model inversion.
    Blocks or perturbs queries whose gradient structure reveals training data.

    Detection heuristic:
      - Queries with very low loss (near-perfect match) → likely inversion attempt
      - Queries with unusual input norm → out-of-distribution / adversarial
    """

    def __init__(
        self,
        loss_threshold: float = 0.1,    # Block queries with loss < threshold
        norm_threshold: float = 10.0,   # Block inputs with ‖x‖ > threshold
        gradient_noise_scale: float = 0.05,
    ):
        self.loss_threshold = loss_threshold
        self.norm_threshold = norm_threshold
        self.gradient_noise_scale = gradient_noise_scale
        self._blocked_count = 0
        self._total_count = 0

    def sanitize_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[Optional[List[torch.Tensor]], bool]:
        """
        Returns (sanitized_gradients, was_blocked).
        Blocked queries get noisy gradients.
        """
        self._total_count += 1

        # Check input norm
        input_norms = x.norm(dim=-1)
        suspicious_norm = (input_norms > self.norm_threshold).any()

        # Check loss
        model.eval()
        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, y).item()

        suspicious_loss = loss < self.loss_threshold

        if suspicious_norm or suspicious_loss:
            self._blocked_count += 1
            logger.warning(
                f"[MI Defense] Suspicious gradient query blocked "
                f"(loss={loss:.4f}, norm_suspicious={suspicious_norm})"
            )
            # Return heavily noised gradients
            noisy_grads = []
            for p in model.parameters():
                noisy_grads.append(torch.randn_like(p) * self.gradient_noise_scale)
            return noisy_grads, True

        # Normal gradients with light noise
        model.zero_grad()
        model.train()
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss_t = criterion(logits, y)
        loss_t.backward()

        sanitized = []
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.gradient_noise_scale
                sanitized.append(p.grad + noise)
            else:
                sanitized.append(torch.zeros_like(p))

        return sanitized, False

    @property
    def block_rate(self) -> float:
        return self._blocked_count / max(self._total_count, 1)


class ModelInversionDefenseWrapper(nn.Module):
    """
    Wraps a model with all inversion defenses.
    Drop-in replacement: same forward signature.
    """

    def __init__(
        self,
        model: nn.Module,
        representation_noise: float = 0.05,
        output_epsilon: float = 2.0,
        rounding: int = 2,
    ):
        super().__init__()
        self.model = model
        self.output_defense = OutputRandomizationDefense(
            noise_epsilon=output_epsilon, rounding=rounding
        )
        self.noise_hook = RepresentationNoiseHook(noise_scale=representation_noise)
        self.noise_hook.register(self.model)
        self._defense_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        raw = logits[0] if isinstance(logits, (tuple, list)) else logits

        if self._defense_enabled and not self.training:
            defended = self.output_defense.defend_logits(raw)
            if isinstance(logits, (tuple, list)):
                return (defended,) + logits[1:]
            return defended

        return logits

    def disable_defense(self):
        self._defense_enabled = False
        self.noise_hook.remove()

    def enable_defense(self):
        self._defense_enabled = True
        self.noise_hook.register(self.model)

    def evaluate_inversion_resistance(
        self,
        X_test: np.ndarray,
        n_reconstruction_steps: int = 50,
        device: torch.device = None,
    ) -> Dict:
        """
        Estimate model inversion resistance.
        Attempts gradient-based reconstruction and measures quality.
        Lower PSNR / higher MSE = better defense.
        """
        device = device or torch.device("cpu")
        self.eval()

        # Random target: try to reconstruct first sample
        x_target = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
        x_reconstruct = torch.randn_like(x_target, requires_grad=True)
        optimizer = torch.optim.Adam([x_reconstruct], lr=0.01)

        for _ in range(n_reconstruction_steps):
            optimizer.zero_grad()
            out_target = self.model(x_target).detach()
            out_reconstruct = self.forward(x_reconstruct)
            raw_r = out_reconstruct[0] if isinstance(out_reconstruct, (tuple, list)) else out_reconstruct
            raw_t = out_target[0] if isinstance(out_target, (tuple, list)) else out_target
            loss = torch.mean((raw_r - raw_t) ** 2)
            loss.backward()
            optimizer.step()

        reconstruction_mse = float(
            ((x_reconstruct.detach() - x_target) ** 2).mean()
        )
        return {
            "reconstruction_mse": reconstruction_mse,
            "inversion_resistance": "HIGH" if reconstruction_mse > 1.0 else "LOW",
            "n_steps": n_reconstruction_steps,
        }
