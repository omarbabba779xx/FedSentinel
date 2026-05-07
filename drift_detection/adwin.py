"""
Concept Drift Detection for FL using ADWIN (ADaptive WINdowing).
Detects when the data distribution shifts (new attack types appear).
Triggers FL re-training round when drift is detected.

ADWIN principle:
  Maintains a variable-size window W.
  Splits W into W0 (old) and W1 (new).
  If |mean(W0) - mean(W1)| > threshold → DRIFT DETECTED → shrink window.

Reference: Bifet & Gavalda "Learning from Time-Changing Data with Adaptive Windowing" (SDM 2007)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from collections import deque
import threading
from utils.logger import get_logger

logger = get_logger("DriftDetection")


class ADWIN:
    """
    ADWIN (Adaptive Windowing) algorithm for concept drift detection.
    Monitors a stream of values and detects distribution shifts.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5):
        self.delta = delta          # confidence parameter (lower = more sensitive)
        self.max_buckets = max_buckets
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._buckets: deque = deque()  # list of (count, sum) pairs
        self.drift_detected = False
        self.n_detections = 0

    def add_element(self, value: float) -> bool:
        """
        Add observation to window. Returns True if drift detected.
        """
        self._width += 1
        self._buckets.append([1, value])
        self._total += value

        self._compress_buckets()

        drift = self._detect_change()
        if drift:
            self.drift_detected = True
            self.n_detections += 1
            logger.warning(f"[ADWIN] Concept DRIFT detected! (n_detections={self.n_detections})")
        else:
            self.drift_detected = False

        return drift

    def _compress_buckets(self):
        """Merge old buckets to maintain O(log n) memory."""
        bucket_count: Dict[int, int] = {}
        for i, (count, _) in enumerate(self._buckets):
            bucket_count[count] = bucket_count.get(count, 0) + 1
            if bucket_count[count] > self.max_buckets:
                if i + 1 < len(self._buckets):
                    b1 = self._buckets[i]
                    b2 = self._buckets[i + 1]
                    new_count = b1[0] + b2[0]
                    new_sum = b1[1] + b2[1]
                    buckets_list = list(self._buckets)
                    buckets_list[i] = [new_count, new_sum]
                    del buckets_list[i + 1]
                    self._buckets = deque(buckets_list)
                break

    def _detect_change(self) -> bool:
        """Check if any split of the window shows significant mean difference."""
        n = self._width
        if n < 2:
            return False

        mean_all = self._total / n
        buckets_list = list(self._buckets)

        cumsum = 0.0
        cumcount = 0

        for bucket in buckets_list[:-1]:
            cumcount += bucket[0]
            cumsum += bucket[1]

            if cumcount == 0 or cumcount == n:
                continue

            mean_left = cumsum / cumcount
            mean_right = (self._total - cumsum) / (n - cumcount)

            n0, n1 = cumcount, n - cumcount
            epsilon_cut = np.sqrt(
                (1.0 / (2.0 * min(n0, n1))) * np.log(4.0 * n / self.delta)
            )

            if abs(mean_left - mean_right) >= epsilon_cut:
                # Remove old data (left part of window)
                self._total -= cumsum
                self._width -= cumcount
                self._buckets = deque(list(self._buckets)[len(list(self._buckets)) - len(buckets_list) + buckets_list.index(bucket) + 1:])
                return True

        return False

    @property
    def window_mean(self) -> float:
        return self._total / max(self._width, 1)

    @property
    def window_size(self) -> int:
        return self._width

    def reset(self):
        self._total = 0.0
        self._width = 0
        self._buckets = deque()
        self.drift_detected = False


class FedDriftMonitor:
    """
    Federated drift detection.
    Each client monitors its local accuracy stream with ADWIN.
    When majority of clients detect drift → server triggers re-training.
    """

    def __init__(
        self,
        num_clients: int,
        drift_threshold: float = 0.5,    # fraction of clients needed to trigger
        delta: float = 0.002,
        on_drift_detected: Optional[Callable] = None,
    ):
        self.num_clients = num_clients
        self.drift_threshold = drift_threshold
        self.on_drift_detected = on_drift_detected
        self._detectors = {i: ADWIN(delta=delta) for i in range(num_clients)}
        self._lock = threading.Lock()
        self._drift_history: List[Dict] = []
        self._global_round = 0

    def update_client(self, client_id: int, accuracy: float) -> bool:
        """Update client's ADWIN with latest accuracy. Returns True if drift."""
        with self._lock:
            return self._detectors[client_id].add_element(accuracy)

    def check_global_drift(self, round_num: int) -> Dict:
        """Check if enough clients detected drift to trigger re-training."""
        self._global_round = round_num
        with self._lock:
            drifted_clients = [cid for cid, det in self._detectors.items() if det.drift_detected]
            drift_fraction = len(drifted_clients) / self.num_clients

            triggered = drift_fraction >= self.drift_threshold
            report = {
                "round": round_num,
                "drift_fraction": drift_fraction,
                "drifted_clients": drifted_clients,
                "global_drift_triggered": triggered,
                "client_window_sizes": {cid: det.window_size for cid, det in self._detectors.items()},
                "client_means": {cid: det.window_mean for cid, det in self._detectors.items()},
            }

            if triggered:
                logger.warning(
                    f"[Round {round_num}] GLOBAL DRIFT triggered! "
                    f"{len(drifted_clients)}/{self.num_clients} clients drifted"
                )
                self._drift_history.append(report)
                if self.on_drift_detected:
                    self.on_drift_detected(report)
                # Reset detectors after global drift response
                for det in self._detectors.values():
                    det.reset()

        return report

    def get_drift_summary(self) -> Dict:
        return {
            "total_global_drifts": len(self._drift_history),
            "drift_rounds": [r["round"] for r in self._drift_history],
            "history": self._drift_history,
        }
