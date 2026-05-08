"""
Shapley Value-Based Incentive Mechanism for Federated Learning.
Fairly rewards clients based on their marginal contribution to model performance.

Shapley value: phi_i = sum_{S subset N\\{i}} [|S|!(|N|-|S|-1)!/|N|!] * [v(S union {i}) - v(S)]

Approximations used (exact is 2^N):
  - Monte Carlo: permutation sampling (O(T*N))
  - Truncated MC: skip coalitions where marginal < threshold (GTG-Shapley)
  - Group Testing: O(N log N) via binary splitting

Reference: Wang et al. "Measure Contribution of Participants in FL" (ICDCS 2020)
           Ghorbani & Zou "Data Shapley" (ICML 2019)
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from itertools import combinations
import threading
import time
from utils.logger import get_logger

logger = get_logger("ShapleyIncentive")


class ShapleyCalculator:
    """
    Computes Shapley values for FL clients based on model performance.
    Uses Monte Carlo approximation with truncation for scalability.
    """

    def __init__(
        self,
        num_clients: int,
        val_function: Callable[[List[int]], float],
        mc_iterations: int = 200,
        truncation_threshold: float = 0.01,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.val_function = val_function   # v(S) → performance metric
        self.mc_iterations = mc_iterations
        self.truncation_threshold = truncation_threshold
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[frozenset, float] = {}
        self._v_grand = None   # v(N) cached

    def _cached_val(self, coalition: List[int]) -> float:
        key = frozenset(coalition)
        if key not in self._cache:
            self._cache[key] = self.val_function(coalition)
        return self._cache[key]

    def compute_exact(self) -> Dict[int, float]:
        """Exact Shapley (feasible only for N ≤ 15)."""
        n = self.num_clients
        shapley = {i: 0.0 for i in range(n)}

        for i in range(n):
            others = [j for j in range(n) if j != i]
            for size in range(len(others) + 1):
                weight = (
                    np.math.factorial(size)
                    * np.math.factorial(n - size - 1)
                    / np.math.factorial(n)
                )
                for subset in combinations(others, size):
                    s_list = list(subset)
                    v_with = self._cached_val(s_list + [i])
                    v_without = self._cached_val(s_list)
                    shapley[i] += weight * (v_with - v_without)

        return shapley

    def compute_monte_carlo(self) -> Dict[int, float]:
        """
        Monte Carlo Shapley approximation via random permutation sampling.
        Truncated: skip remaining clients in permutation if marginal < threshold.
        """
        n = self.num_clients
        marginals = {i: [] for i in range(n)}

        v_grand = self._cached_val(list(range(n)))
        v_empty = self._cached_val([])

        for _ in range(self.mc_iterations):
            perm = self._rng.permutation(n).tolist()
            coalition = []
            v_prev = v_empty
            truncated = False

            for client_id in perm:
                if truncated:
                    marginals[client_id].append(0.0)
                    continue

                coalition_with = coalition + [client_id]
                v_curr = self._cached_val(coalition_with)
                marginal = v_curr - v_prev

                marginals[client_id].append(marginal)
                coalition = coalition_with
                v_prev = v_curr

                # GTG-Shapley truncation: stop if close to grand coalition value
                if abs(v_grand - v_curr) < self.truncation_threshold:
                    truncated = True

        shapley = {i: float(np.mean(marginals[i])) for i in range(n)}
        return shapley

    def compute_group_testing(self) -> Dict[int, float]:
        """
        Group testing approximation: O(N log N) queries via binary splitting.
        Less accurate but extremely fast for large N.
        """
        n = self.num_clients
        v_grand = self._cached_val(list(range(n)))
        v_empty = self._cached_val([])
        total_gain = v_grand - v_empty

        shapley = self._binary_split(list(range(n)), total_gain)
        return shapley

    def _binary_split(self, clients: List[int], total: float) -> Dict[int, float]:
        if len(clients) == 1:
            return {clients[0]: total}
        if len(clients) == 0:
            return {}

        mid = len(clients) // 2
        left, right = clients[:mid], clients[mid:]

        v_left = self._cached_val(left)
        v_right = self._cached_val(right)
        v_base = self._cached_val([])

        # Distribute proportionally by individual contribution
        left_gain = v_left - v_base
        right_gain = v_right - v_base

        total_individual = left_gain + right_gain
        if total_individual == 0:
            left_share = total / 2
            right_share = total / 2
        else:
            left_share = total * left_gain / total_individual
            right_share = total * right_gain / total_individual

        result = {}
        result.update(self._binary_split(left, left_share))
        result.update(self._binary_split(right, right_share))
        return result

    def normalize(self, shapley: Dict[int, float]) -> Dict[int, float]:
        """Normalize Shapley values to [0, 1] range."""
        vals = np.array(list(shapley.values()))
        min_v, max_v = vals.min(), vals.max()
        if max_v == min_v:
            return {k: 1.0 / len(shapley) for k in shapley}
        return {k: float((v - min_v) / (max_v - min_v)) for k, v in shapley.items()}


class FedShapleyIncentive:
    """
    Federated Shapley incentive system.
    Tracks per-client contributions across rounds and adjusts:
      - Participation probability (higher Shapley → more likely selected)
      - Reward score for external incentive distribution
      - Data quality weight for aggregation
    """

    def __init__(
        self,
        num_clients: int,
        method: str = "monte_carlo",   # "exact" | "monte_carlo" | "group_testing"
        mc_iterations: int = 100,
        history_window: int = 10,      # rounds to average
        selection_temp: float = 2.0,   # softmax temperature for client selection
    ):
        self.num_clients = num_clients
        self.method = method
        self.mc_iterations = mc_iterations
        self.history_window = history_window
        self.selection_temp = selection_temp

        self._shapley_history: List[Dict[int, float]] = []
        self._cumulative: Dict[int, float] = {i: 0.0 for i in range(num_clients)}
        self._round_count = 0
        self._lock = threading.Lock()

    def compute_round_shapley(
        self,
        client_models: Dict[int, np.ndarray],
        val_fn: Callable[[List[int]], float],
    ) -> Dict[int, float]:
        """
        Compute Shapley values for this round.
        val_fn(coalition) should return accuracy/F1 of model aggregated from coalition.
        """
        calc = ShapleyCalculator(
            num_clients=self.num_clients,
            val_function=val_fn,
            mc_iterations=self.mc_iterations,
        )

        if self.method == "exact" and self.num_clients <= 12:
            shapley = calc.compute_exact()
        elif self.method == "group_testing":
            shapley = calc.compute_group_testing()
        else:
            shapley = calc.compute_monte_carlo()

        normalized = calc.normalize(shapley)

        with self._lock:
            self._shapley_history.append(normalized)
            if len(self._shapley_history) > self.history_window:
                self._shapley_history.pop(0)

            for i in range(self.num_clients):
                self._cumulative[i] += normalized.get(i, 0.0)

            self._round_count += 1

        logger.info(f"Shapley values round {self._round_count}: {normalized}")
        return normalized

    def get_selection_probabilities(self) -> np.ndarray:
        """Softmax over recent average Shapley → client selection weights."""
        recent = self._shapley_history[-self.history_window:]
        if not recent:
            return np.ones(self.num_clients) / self.num_clients

        avg = np.zeros(self.num_clients)
        for r in recent:
            for i in range(self.num_clients):
                avg[i] += r.get(i, 0.0)
        avg /= len(recent)

        exp_vals = np.exp(avg * self.selection_temp)
        return exp_vals / exp_vals.sum()

    def select_clients(self, num_to_select: int) -> List[int]:
        """Sample clients weighted by Shapley scores."""
        probs = self.get_selection_probabilities()
        selected = np.random.choice(
            self.num_clients, size=num_to_select, replace=False, p=probs
        )
        return selected.tolist()

    def get_aggregation_weights(self) -> Dict[int, float]:
        """Weight each client's model by their normalized Shapley value during aggregation."""
        recent = self._shapley_history[-1] if self._shapley_history else {i: 1.0 for i in range(self.num_clients)}
        total = sum(recent.values()) or 1.0
        return {i: recent.get(i, 0.0) / total for i in range(self.num_clients)}

    def get_reward_report(self) -> Dict:
        """Return full reward summary for external distribution."""
        with self._lock:
            probs = self.get_selection_probabilities()
            return {
                "round": self._round_count,
                "cumulative_scores": dict(self._cumulative),
                "selection_probabilities": {i: float(probs[i]) for i in range(self.num_clients)},
                "recent_shapley": self._shapley_history[-1] if self._shapley_history else {},
                "top_contributor": int(max(self._cumulative, key=self._cumulative.get)),
                "history_window": self.history_window,
            }

    def flag_free_riders(self, threshold: float = 0.05) -> List[int]:
        """Clients with normalized Shapley < threshold are suspected free-riders."""
        recent = self._shapley_history[-1] if self._shapley_history else {}
        return [i for i, v in recent.items() if v < threshold]
