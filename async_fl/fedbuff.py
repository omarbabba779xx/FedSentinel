"""
Asynchronous Federated Learning — FedBuff algorithm.
Server aggregates whenever buffer reaches capacity, no waiting for slow clients.
Stale gradients scaled by 1/staleness to reduce bias.

Reference: Nguyen et al. "FEDERATED LEARNING WITH BUFFERED ASYNCHRONOUS AGGREGATION" (AISTATS 2022)
"""

import threading
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("FedBuff")


@dataclass
class ClientUpdate:
    client_id: int
    weights: List[np.ndarray]
    num_samples: int
    round_submitted: int          # which global round the client started from
    timestamp: float = field(default_factory=time.time)
    metrics: dict = field(default_factory=dict)

    @property
    def staleness(self) -> int:
        """How many global rounds behind this update is."""
        return 0  # set externally by server


class FedBuffServer:
    """
    Asynchronous FL server with buffered aggregation.

    Key differences from FedAvg:
    - Clients train independently in parallel
    - Server aggregates when buffer_size updates arrive (no waiting for all)
    - Stale updates weighted by α(τ) = 1 / (1 + staleness)
    - Supports concurrent client threads
    """

    def __init__(
        self,
        initial_weights: List[np.ndarray],
        buffer_size: int = 3,
        num_rounds: int = 50,
        staleness_fn: str = "reciprocal",   # reciprocal | constant | polynomial
        aggregation: str = "fedavg",
        on_round_complete: Optional[Callable] = None,
    ):
        self.global_weights = [w.copy() for w in initial_weights]
        self.buffer_size = buffer_size
        self.num_rounds = num_rounds
        self.staleness_fn = staleness_fn
        self.aggregation = aggregation
        self.on_round_complete = on_round_complete

        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._round = 0
        self._history: List[dict] = []
        self._total_updates = 0

    def _staleness_weight(self, staleness: int) -> float:
        if self.staleness_fn == "reciprocal":
            return 1.0 / (1.0 + staleness)
        elif self.staleness_fn == "polynomial":
            return 1.0 / ((1.0 + staleness) ** 0.5)
        else:
            return 1.0

    def submit_update(self, update: ClientUpdate):
        """Thread-safe submission of client update."""
        with self._lock:
            update.staleness = self._round - update.round_submitted
            self._buffer.append(update)
            self._total_updates += 1
            logger.info(
                f"Client {update.client_id} submitted update "
                f"(staleness={update.staleness}, buffer={len(self._buffer)}/{self.buffer_size})"
            )

            if len(self._buffer) >= self.buffer_size:
                self._aggregate_buffer()

    def _aggregate_buffer(self):
        """Drain buffer and aggregate with staleness weighting."""
        updates = [self._buffer.popleft() for _ in range(min(self.buffer_size, len(self._buffer)))]
        self._round += 1

        weights_list = [u.weights for u in updates]
        sample_counts = [u.num_samples for u in updates]
        staleness_weights = [self._staleness_weight(u.staleness) for u in updates]

        # Combined weight: num_samples * staleness_weight
        combined = [s * sw for s, sw in zip(sample_counts, staleness_weights)]
        total = sum(combined)

        aggregated = [
            sum(weights_list[i][j] * combined[i] / total for i in range(len(updates)))
            for j in range(len(self.global_weights))
        ]
        self.global_weights = aggregated

        round_record = {
            "round": self._round,
            "clients": [u.client_id for u in updates],
            "avg_staleness": float(np.mean([u.staleness for u in updates])),
            "staleness_weights": staleness_weights,
        }
        self._history.append(round_record)

        logger.info(
            f"[AsyncFL Round {self._round}] Aggregated {len(updates)} updates | "
            f"avg_staleness={round_record['avg_staleness']:.1f}"
        )

        if self.on_round_complete:
            self.on_round_complete(self._round, self.global_weights, round_record)

    def get_global_weights(self) -> Tuple[List[np.ndarray], int]:
        """Return current global weights + current round (thread-safe)."""
        with self._lock:
            return [w.copy() for w in self.global_weights], self._round

    @property
    def current_round(self) -> int:
        return self._round

    def get_history(self) -> List[dict]:
        return self._history.copy()


class AsyncFLClient(threading.Thread):
    """
    Async FL client thread. Trains independently, submits to server buffer.
    """

    def __init__(
        self,
        client_id: int,
        server: FedBuffServer,
        train_fn: Callable,         # fn(weights, client_id) -> (new_weights, num_samples, metrics)
        num_local_rounds: int = 5,
        sleep_between_rounds: float = 0.0,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon)
        self.client_id = client_id
        self.server = server
        self.train_fn = train_fn
        self.num_local_rounds = num_local_rounds
        self.sleep_between_rounds = sleep_between_rounds
        self.logger = get_logger(f"AsyncClient-{client_id}")

    def run(self):
        for _ in range(self.num_local_rounds):
            global_weights, current_round = self.server.get_global_weights()
            self.logger.info(f"Starting local training from round {current_round}")

            new_weights, num_samples, metrics = self.train_fn(global_weights, self.client_id)

            update = ClientUpdate(
                client_id=self.client_id,
                weights=new_weights,
                num_samples=num_samples,
                round_submitted=current_round,
                metrics=metrics,
            )
            self.server.submit_update(update)

            if self.sleep_between_rounds > 0:
                time.sleep(self.sleep_between_rounds)


def run_async_fl(
    initial_weights: List[np.ndarray],
    train_fns: List[Callable],
    buffer_size: int = 3,
    num_rounds: int = 10,
    staleness_fn: str = "reciprocal",
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    High-level async FL simulation.
    train_fns[i](weights, client_id) → (new_weights, num_samples, metrics)
    """
    server = FedBuffServer(
        initial_weights=initial_weights,
        buffer_size=buffer_size,
        num_rounds=num_rounds,
        staleness_fn=staleness_fn,
    )

    clients = [
        AsyncFLClient(
            client_id=i,
            server=server,
            train_fn=train_fns[i],
            num_local_rounds=num_rounds,
            sleep_between_rounds=np.random.uniform(0, 0.1),
        )
        for i in range(len(train_fns))
    ]

    for c in clients:
        c.start()
    for c in clients:
        c.join(timeout=300)

    final_weights, _ = server.get_global_weights()
    return final_weights, server.get_history()
