"""
Privacy budget accountant using Rényi DP + Gaussian DP (f-DP) dual accounting.
Tracks cumulative epsilon across FL rounds with tight bounds.

Dual accounting strategy:
  1. RDP (Mironov 2017) — tight for composition, converted via optimal RDP→(ε,δ) conversion
  2. GDP (Dong et al. 2022) — f-DP via CLT: σ_GDP = σ/sensitivity, μ = √(2T) / σ_GDP
     Convert GDP→(ε,δ): ε = μ² / 2 + μ*√(2 ln(1/δ)), tight for large T

Final ε = min(RDP bound, GDP bound) — take whichever is tighter.

References:
  Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
  Dong, Roth, Su (2022) "Gaussian Differential Privacy"
  Balle et al. (2020) "Hypothesis Testing Interpretations and Renyi DP"
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.special import erfc
from utils.logger import get_logger

logger = get_logger("PrivacyAccountant")

ORDERS = list(range(2, 128)) + [256, 512, 1024]


# ─── RDP accounting ──────────────────────────────────────────────────────────

def _rdp_gaussian_per_step(alpha: float, noise_multiplier: float) -> float:
    """Exact RDP for Gaussian mechanism: ε_R(α) = α / (2σ²)."""
    if noise_multiplier == 0:
        return np.inf
    return alpha / (2.0 * noise_multiplier ** 2)


def _rdp_subsampled_poisson(alpha: int, noise_multiplier: float, sample_rate: float) -> float:
    """
    Tight RDP for Poisson subsampling (Wang et al. 2019 / Mironov et al. 2019).
    Uses the log-sum-exp bound for integer alpha.
    """
    if noise_multiplier == 0:
        return np.inf
    if sample_rate >= 1.0:
        return _rdp_gaussian_per_step(alpha, noise_multiplier)

    q = sample_rate
    sigma = noise_multiplier

    if alpha == 1:
        # KL divergence bound
        return q * (np.exp(1.0 / sigma ** 2) - 1)

    # Binomial theorem expansion (tightest known bound)
    log_terms = []
    for k in range(0, alpha + 1):
        # log C(alpha, k) * q^k * (1-q)^(alpha-k)
        log_binom = (
            _log_comb(alpha, k)
            + k * np.log(q)
            + (alpha - k) * np.log(1.0 - q)
        )
        # Gaussian RDP at order k
        if k >= 2:
            rdp_k = k * (k - 1) / (2.0 * sigma ** 2)
        else:
            rdp_k = 0.0

        log_terms.append(log_binom + (alpha - 1) * rdp_k)

    log_sum = _log_sum_exp(log_terms)
    return log_sum / (alpha - 1)


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) via Stirling / direct for small n."""
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _log_sum_exp(log_vals: List[float]) -> float:
    arr = np.array(log_vals)
    m = arr.max()
    return m + np.log(np.sum(np.exp(arr - m)))


def _compute_rdp(noise_multiplier: float, sample_rate: float, steps: int) -> np.ndarray:
    rdp = np.zeros(len(ORDERS))
    for i, alpha in enumerate(ORDERS):
        rdp[i] = _rdp_subsampled_poisson(int(alpha), noise_multiplier, sample_rate)
    return rdp * steps


def _rdp_to_dp(rdp: np.ndarray, delta: float) -> float:
    """
    Tight RDP → (ε, δ)-DP conversion (Balle et al. 2020 / Canonne et al. 2020).
    ε = min over α of: rdp[α] + log((α-1)/α) - (log δ + log(α-1)) / (α-1)
    """
    eps_candidates = []
    for i, alpha in enumerate(ORDERS):
        if np.isinf(rdp[i]) or alpha <= 1:
            continue
        # Standard RDP→DP conversion
        eps1 = rdp[i] + np.log((alpha - 1) / alpha) - (np.log(delta) + np.log(alpha - 1)) / (alpha - 1)
        # Improved conversion (Balle et al. Theorem 21)
        eps2 = rdp[i] + np.log(1 - 1.0 / alpha) - np.log(delta) / (alpha - 1)
        eps_candidates.append(min(eps1, eps2))

    return float(min(eps_candidates)) if eps_candidates else np.inf


# ─── GDP (f-DP) accounting ───────────────────────────────────────────────────

def _gdp_mu_from_rounds(noise_multiplier: float, sample_rate: float, steps: int) -> float:
    """
    Central Limit Theorem composition for GDP.
    Each Gaussian mechanism is (μ₀)-GDP with μ₀ = sensitivity/σ = 1/σ.
    Subsampled by q: effective μ₀ ≈ √2 * q / σ (Dong et al. Lemma 4.5).
    CLT: after T steps, total μ = √T * effective_μ₀.
    """
    sigma = noise_multiplier
    if sigma == 0:
        return np.inf
    # Subsampled single-step GDP parameter
    mu_per_step = sample_rate * np.sqrt(2.0) / sigma
    # CLT composition
    return mu_per_step * np.sqrt(steps)


def _gdp_to_dp(mu: float, delta: float) -> float:
    """
    Convert GDP μ to (ε, δ)-DP via tight analytic formula.
    ε(δ) = μ²/2 + μ * Φ⁻¹(1-δ) where Φ⁻¹ is inverse normal CDF.
    Equivalently: ε = Φ(μ/2 - Φ⁻¹(δ)/μ) * ... (see Dong et al. Thm 3.2).
    Numerical inversion via binary search.
    """
    if mu == 0:
        return 0.0
    if np.isinf(mu):
        return np.inf

    # Tight formula: δ(ε) = Φ(-ε/μ + μ/2) - exp(ε) * Φ(-ε/μ - μ/2)
    # We binary-search ε such that δ(ε) = delta
    def delta_from_eps(eps: float) -> float:
        from scipy.stats import norm
        a = -eps / mu + mu / 2.0
        b = -eps / mu - mu / 2.0
        return float(norm.cdf(a) - np.exp(eps) * norm.cdf(b))

    low, high = 0.0, mu ** 2  # ε ≤ μ² for reasonable δ
    # Expand upper bound if needed
    while delta_from_eps(high) > delta:
        high *= 2.0
        if high > 1000:
            return high

    for _ in range(80):
        mid = (low + high) / 2.0
        if delta_from_eps(mid) > delta:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


# ─── Main accountant ─────────────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Dual RDP + GDP privacy accountant.
    Reports min(ε_RDP, ε_GDP) — the tightest bound at each round.
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self._rdp_accum: np.ndarray = np.zeros(len(ORDERS))
        self._gdp_mu_sq_accum: float = 0.0   # accumulate μ² for CLT
        self._round_history: List[dict] = []
        self.total_rounds = 0

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> Tuple[float, bool]:
        """Account for one FL round. Returns (current_epsilon, budget_exceeded)."""
        # RDP path
        rdp_step = _compute_rdp(noise_multiplier, sample_rate, num_steps)
        self._rdp_accum += rdp_step
        eps_rdp = _rdp_to_dp(self._rdp_accum, self.target_delta)

        # GDP path (CLT: accumulate μ² then take sqrt)
        mu_step = _gdp_mu_from_rounds(noise_multiplier, sample_rate, num_steps)
        self._gdp_mu_sq_accum += mu_step ** 2
        mu_total = np.sqrt(self._gdp_mu_sq_accum)
        eps_gdp = _gdp_to_dp(mu_total, self.target_delta)

        # Take the tighter bound
        current_eps = float(min(eps_rdp, eps_gdp))
        budget_exceeded = current_eps > self.target_epsilon

        self.total_rounds += 1
        record = {
            "round": self.total_rounds,
            "epsilon": current_eps,
            "epsilon_rdp": float(eps_rdp),
            "epsilon_gdp": float(eps_gdp),
            "gdp_mu": float(mu_total),
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "budget_exceeded": budget_exceeded,
        }
        self._round_history.append(record)

        if budget_exceeded:
            logger.warning(
                f"[Round {self.total_rounds}] Budget EXCEEDED: "
                f"ε={current_eps:.4f} (RDP={eps_rdp:.4f}, GDP={eps_gdp:.4f}) > target={self.target_epsilon}"
            )
        else:
            logger.info(
                f"[Round {self.total_rounds}] ε={current_eps:.4f}/{self.target_epsilon} "
                f"(RDP={eps_rdp:.4f}, GDP={eps_gdp:.4f}, μ={mu_total:.4f})"
            )

        return current_eps, budget_exceeded

    @property
    def current_epsilon(self) -> float:
        eps_rdp = _rdp_to_dp(self._rdp_accum, self.target_delta)
        mu = np.sqrt(self._gdp_mu_sq_accum)
        eps_gdp = _gdp_to_dp(mu, self.target_delta)
        return float(min(eps_rdp, eps_gdp))

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.target_epsilon - self.current_epsilon)

    def get_report(self) -> dict:
        eps = self.current_epsilon
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": eps,
            "remaining_budget": max(0.0, self.target_epsilon - eps),
            "total_rounds": self.total_rounds,
            "budget_exceeded": eps > self.target_epsilon,
            "accounting_method": "min(RDP, GDP)",
            "history": self._round_history,
        }

    def reset(self):
        self._rdp_accum = np.zeros(len(ORDERS))
        self._gdp_mu_sq_accum = 0.0
        self._round_history = []
        self.total_rounds = 0


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    tolerance: float = 0.01,
) -> float:
    """Binary search for σ achieving target (ε, δ) using dual accounting."""
    low, high = 0.1, 100.0

    for _ in range(100):
        mid = (low + high) / 2.0
        acct = PrivacyAccountant(target_epsilon=target_epsilon, target_delta=target_delta)
        acct.step(mid, sample_rate, steps)
        eps = acct.current_epsilon

        if eps < target_epsilon - tolerance:
            high = mid
        elif eps > target_epsilon + tolerance:
            low = mid
        else:
            return mid

    return (low + high) / 2.0
