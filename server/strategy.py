"""
Custom Flower strategy combining:
- FedAvg / FedProx base
- Byzantine-robust aggregation (pluggable)
- Free-rider detection
- Privacy accounting
- Per-round metric logging
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from functools import reduce
import flwr as fl
from flwr.common import (
    Parameters, FitRes, EvaluateRes, FitIns, EvaluateIns,
    Scalar, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from defense import (
    krum, multi_krum, trimmed_mean, coordinate_median,
    flame, fltrust, FreeRiderDetector,
)
from privacy import PrivacyAccountant
from utils.logger import get_logger
from utils.helpers import save_json


class FedSentinelStrategy(fl.server.strategy.Strategy):
    """
    FedSentinel aggregation strategy with full Byzantine robustness.
    """

    def __init__(
        self,
        aggregation: str = "fedavg",
        num_byzantine: int = 1,
        trimmed_mean_beta: float = 0.1,
        flame_noise_sigma: float = 0.001,
        fedprox_mu: float = 0.01,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        dp_enabled: bool = True,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        free_rider_detection: bool = True,
        results_path: str = "./results/training_history.json",
        initial_parameters: Optional[Parameters] = None,
    ):
        self.aggregation = aggregation
        self.num_byzantine = num_byzantine
        self.trimmed_mean_beta = trimmed_mean_beta
        self.flame_noise_sigma = flame_noise_sigma
        self.fedprox_mu = fedprox_mu
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.results_path = results_path
        self.initial_parameters = initial_parameters
        self.logger = get_logger("FedSentinelStrategy")

        self.privacy_accountant = PrivacyAccountant(target_epsilon, target_delta) if dp_enabled else None
        self.free_rider_detector = FreeRiderDetector() if free_rider_detection else None

        self._round = 0
        self._history: List[dict] = []
        self._global_weights: Optional[List[np.ndarray]] = None

    # ─── Required strategy interface ────────────────────

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=max(self.min_fit_clients, int(client_manager.num_available() * self.fraction_fit)),
            min_num_clients=self.min_fit_clients,
        )
        fit_ins = FitIns(parameters, {"server_round": server_round, "local_epochs": 3})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._round = server_round

        if not results:
            return None, {}

        weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_samples_list = [fit_res.num_examples for _, fit_res in results]
        client_metrics = [fit_res.metrics for _, fit_res in results]

        # Free-rider detection
        rejected_free_riders = []
        if self.free_rider_detector is not None and self._global_weights is not None:
            filtered_weights, honest_idx, suspect_idx = self.free_rider_detector.filter_free_riders(
                weights_list, self._global_weights
            )
            if suspect_idx:
                self.logger.warning(f"[Round {server_round}] Free-riders detected: clients {suspect_idx}")
                weights_list = filtered_weights
                num_samples_list = [num_samples_list[i] for i in honest_idx]
                client_metrics = [client_metrics[i] for i in honest_idx]
                rejected_free_riders = suspect_idx

        # Aggregation
        aggregated, agg_info = self._aggregate(weights_list, num_samples_list)
        self._global_weights = aggregated

        # Privacy accounting
        epsilon = 0.0
        if self.privacy_accountant is not None:
            sample_rate = sum(num_samples_list) / max(sum(num_samples_list), 1)
            epsilon, _ = self.privacy_accountant.step(
                noise_multiplier=1.1, sample_rate=0.1, num_steps=len(weights_list)
            )

        # Aggregate metrics
        avg_acc = float(np.mean([m.get("val_accuracy", 0) for m in client_metrics]))
        avg_loss = float(np.mean([m.get("val_loss", 0) for m in client_metrics]))

        round_record = {
            "round": server_round,
            "aggregation": self.aggregation,
            "num_clients": len(weights_list),
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "rejected_free_riders": rejected_free_riders,
            "agg_info": agg_info,
        }
        self._history.append(round_record)
        save_json(self._history, self.results_path)

        self.logger.info(
            f"[Round {server_round}] {self.aggregation.upper()} | "
            f"clients={len(weights_list)} | acc={avg_acc:.4f} | ε={epsilon:.4f}"
        )

        return ndarrays_to_parameters(aggregated), {
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "round": server_round,
        }

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(
            num_clients=max(self.min_evaluate_clients, int(client_manager.num_available() * self.fraction_evaluate)),
            min_num_clients=self.min_evaluate_clients,
        )
        eval_ins = EvaluateIns(parameters, {})
        return [(c, eval_ins) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_samples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_samples
        weighted_acc = sum(r.metrics.get("accuracy", 0) * r.num_examples for _, r in results) / total_samples

        self.logger.info(f"[Round {server_round}] Eval | loss={weighted_loss:.4f} | acc={weighted_acc:.4f}")
        return weighted_loss, {"accuracy": weighted_acc}

    def evaluate(self, server_round: int, parameters: Parameters):
        return None

    # ─── Aggregation dispatch ────────────────────────────

    def _aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
    ) -> Tuple[List[np.ndarray], dict]:
        n = len(weights_list)
        alg = self.aggregation.lower()

        if alg == "fedavg":
            total = sum(num_samples_list)
            agg = [
                sum(weights_list[i][j] * num_samples_list[i] / total for i in range(n))
                for j in range(len(weights_list[0]))
            ]
            return agg, {"algorithm": "fedavg"}

        elif alg == "fedprox":
            total = sum(num_samples_list)
            agg = [
                sum(weights_list[i][j] * num_samples_list[i] / total for i in range(n))
                for j in range(len(weights_list[0]))
            ]
            return agg, {"algorithm": "fedprox", "mu": self.fedprox_mu}

        elif alg == "krum":
            return krum(weights_list, self.num_byzantine)

        elif alg == "multi_krum":
            return multi_krum(weights_list, self.num_byzantine)

        elif alg == "trimmed_mean":
            return trimmed_mean(weights_list, self.trimmed_mean_beta)

        elif alg == "median":
            return coordinate_median(weights_list)

        elif alg == "flame":
            return flame(weights_list, noise_sigma=self.flame_noise_sigma)

        else:
            # Default FedAvg
            total = sum(num_samples_list)
            agg = [
                sum(weights_list[i][j] * num_samples_list[i] / total for i in range(n))
                for j in range(len(weights_list[0]))
            ]
            return agg, {"algorithm": "fedavg_fallback"}
