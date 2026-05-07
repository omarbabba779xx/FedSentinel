"""
Privacy budget accountant using Rényi Differential Privacy (RDP).
Tracks cumulative epsilon across FL rounds.
Based on: Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
"""

import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("PrivacyAccountant")

# Rényi orders to evaluate
ORDERS = list(range(2, 128)) + [256, 512, 1024]


def _compute_rdp_gaussian(noise_multiplier: float, sample_rate: float, steps: int) -> np.ndarray:
    """
    Compute RDP for Gaussian mechanism with subsampling (Poisson).
    Returns RDP values at each order in ORDERS.
    """
    rdp = np.zeros(len(ORDERS))
    for i, alpha in enumerate(ORDERS):
        if noise_multiplier == 0:
            rdp[i] = np.inf
        else:
            # Gaussian mechanism RDP
            rdp_per_step = alpha / (2 * noise_multiplier ** 2)
            # Subsampling amplification (simplified bound)
            if sample_rate < 1:
                log_term = np.log(
                    1 + sample_rate ** 2 * (np.exp(rdp_per_step * (alpha + 1) / alpha) - 1)
                )
                rdp[i] = min(rdp_per_step, log_term / (alpha - 1) if alpha > 1 else rdp_per_step)
            else:
                rdp[i] = rdp_per_step
    return rdp * steps


def _rdp_to_dp(rdp: np.ndarray, delta: float) -> float:
    """Convert RDP to (ε, δ)-DP."""
    eps_values = []
    for i, alpha in enumerate(ORDERS):
        if rdp[i] == np.inf:
            continue
        if alpha == 1:
            eps = rdp[i] + np.log(1 / delta)
        else:
            eps = rdp[i] + np.log((alpha - 1) / alpha) - (np.log(delta) + np.log(alpha - 1)) / (alpha - 1)
        eps_values.append(eps)
    return float(min(eps_values)) if eps_values else np.inf


class PrivacyAccountant:
    """
    Tracks (ε, δ)-DP budget across FL rounds.
    Each round of FL = one call to step().
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self._rdp_history: np.ndarray = np.zeros(len(ORDERS))
        self._round_history: List[dict] = []
        self.total_rounds = 0

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> Tuple[float, bool]:
        """
        Account for one FL round.
        Returns (current_epsilon, budget_exceeded).
        """
        rdp_step = _compute_rdp_gaussian(noise_multiplier, sample_rate, num_steps)
        self._rdp_history += rdp_step

        current_eps = _rdp_to_dp(self._rdp_history, self.target_delta)
        budget_exceeded = current_eps > self.target_epsilon

        self.total_rounds += 1
        self._round_history.append({
            "round": self.total_rounds,
            "epsilon": current_eps,
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "budget_exceeded": budget_exceeded,
        })

        if budget_exceeded:
            logger.warning(f"[Round {self.total_rounds}] Privacy budget exceeded! ε={current_eps:.4f} > target={self.target_epsilon}")
        else:
            logger.info(f"[Round {self.total_rounds}] ε={current_eps:.4f}/{self.target_epsilon} | δ={self.target_delta}")

        return current_eps, budget_exceeded

    @property
    def current_epsilon(self) -> float:
        return _rdp_to_dp(self._rdp_history, self.target_delta)

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.target_epsilon - self.current_epsilon)

    def get_report(self) -> dict:
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": self.current_epsilon,
            "remaining_budget": self.remaining_budget,
            "total_rounds": self.total_rounds,
            "budget_exceeded": self.current_epsilon > self.target_epsilon,
            "history": self._round_history,
        }

    def reset(self):
        self._rdp_history = np.zeros(len(ORDERS))
        self._round_history = []
        self.total_rounds = 0


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    tolerance: float = 0.01,
) -> float:
    """Binary search for noise multiplier achieving target (ε, δ)."""
    low, high = 0.1, 100.0

    for _ in range(100):
        mid = (low + high) / 2.0
        rdp = _compute_rdp_gaussian(mid, sample_rate, steps)
        eps = _rdp_to_dp(rdp, target_delta)

        if eps < target_epsilon - tolerance:
            high = mid
        elif eps > target_epsilon + tolerance:
            low = mid
        else:
            return mid

    return (low + high) / 2.0
