"""
Free-rider detection: identify clients not contributing real computation.
"""

import numpy as np
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger("FreeRiderDetector")


class FreeRiderDetector:
    """
    Detects free-riders by analyzing update magnitudes and cosine similarity
    to the global model over time.

    Methods:
    - delta_check:    near-zero delta → suspect
    - cosine_check:   update too similar to global → suspect
    - historical:     track contribution scores across rounds
    """

    def __init__(
        self,
        delta_threshold: float = 1e-3,
        cosine_threshold: float = 0.999,
        history_window: int = 5,
        min_rounds: int = 3,
    ):
        self.delta_threshold = delta_threshold
        self.cosine_threshold = cosine_threshold
        self.history_window = history_window
        self.min_rounds = min_rounds
        self._client_history: Dict[int, List[float]] = {}
        self._round = 0

    def analyze(
        self,
        client_weights: List[List[np.ndarray]],
        global_weights: List[np.ndarray],
    ) -> Dict[int, dict]:
        self._round += 1
        results = {}

        for cid, weights in enumerate(client_weights):
            delta = [w - gw for w, gw in zip(weights, global_weights)]
            flat_delta = np.concatenate([d.flatten() for d in delta])
            flat_global = np.concatenate([w.flatten() for w in global_weights])

            delta_norm = float(np.linalg.norm(flat_delta))

            global_norm = np.linalg.norm(flat_global)
            cos_sim = float(np.dot(flat_delta, flat_global) / (delta_norm * global_norm + 1e-8)) if delta_norm > 1e-8 else 1.0

            contribution_score = delta_norm / (global_norm + 1e-8)

            if cid not in self._client_history:
                self._client_history[cid] = []
            self._client_history[cid].append(contribution_score)

            history = self._client_history[cid][-self.history_window:]
            avg_contribution = float(np.mean(history))

            is_free_rider = False
            reasons = []

            if delta_norm < self.delta_threshold:
                is_free_rider = True
                reasons.append(f"delta_norm={delta_norm:.6f} < {self.delta_threshold}")

            if cos_sim > self.cosine_threshold:
                is_free_rider = True
                reasons.append(f"cosine_sim={cos_sim:.6f} > {self.cosine_threshold}")

            if len(history) >= self.min_rounds and avg_contribution < self.delta_threshold / 10:
                is_free_rider = True
                reasons.append(f"avg_contribution={avg_contribution:.8f} too low over {len(history)} rounds")

            results[cid] = {
                "delta_norm": delta_norm,
                "cosine_sim": cos_sim,
                "contribution_score": contribution_score,
                "avg_contribution": avg_contribution,
                "is_free_rider": is_free_rider,
                "reasons": reasons,
            }

            if is_free_rider:
                logger.warning(f"[Round {self._round}] Client {cid} suspected FREE-RIDER: {reasons}")

        return results

    def filter_free_riders(
        self,
        client_weights: List[List[np.ndarray]],
        global_weights: List[np.ndarray],
    ) -> Tuple[List[List[np.ndarray]], List[int], List[int]]:
        analysis = self.analyze(client_weights, global_weights)
        honest = [i for i, r in analysis.items() if not r["is_free_rider"]]
        suspects = [i for i, r in analysis.items() if r["is_free_rider"]]

        if not honest:
            logger.warning("All clients flagged as free-riders, using all anyway.")
            honest = list(range(len(client_weights)))
            suspects = []

        return [client_weights[i] for i in honest], honest, suspects
