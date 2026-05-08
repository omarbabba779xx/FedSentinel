"""
Federated Evaluation Protocol.
Clients evaluate locally on their own held-out data → server aggregates.
No test data leaves clients.

Protocol:
  1. Server broadcasts current global model
  2. Each client evaluates on local val set → returns metrics dict
  3. Server aggregates: weighted average by client dataset size
  4. Returns global metric estimate with per-client breakdown

Benefits vs centralized eval:
  - Test data never leaves clients (privacy-preserving evaluation)
  - Captures per-client performance heterogeneity
  - Detects distribution shift per client
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score
from utils.logger import get_logger

logger = get_logger("FederatedEval")

ATTACK_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Dict:
    """Evaluate model on a DataLoader. Returns metrics dict."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / max(len(all_labels), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    per_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[ATTACK_NAMES[c] if c < len(ATTACK_NAMES) else str(c)] = {
                "accuracy": float((all_preds[mask] == c).mean()),
                "count": int(mask.sum()),
            }

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_roc": float(auc),
        "loss": float(avg_loss),
        "num_samples": len(all_labels),
        "per_class": per_class,
    }


class FederatedEvaluator:
    """
    Coordinates federated evaluation across N clients.
    Each client evaluates locally; server aggregates results.
    """

    def __init__(
        self,
        num_classes: int = 5,
        device: torch.device = None,
        min_eval_samples: int = 10,
    ):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.min_eval_samples = min_eval_samples
        self._eval_history: List[Dict] = []

    def client_evaluate(
        self,
        client_id: int,
        model: nn.Module,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 256,
    ) -> Dict:
        """Single client evaluation — runs locally on client."""
        if len(X_val) < self.min_eval_samples:
            logger.warning(f"Client {client_id}: too few samples ({len(X_val)}) for eval")
            return {"client_id": client_id, "skipped": True, "num_samples": len(X_val)}

        dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        metrics = evaluate_on_loader(model, loader, self.device, self.num_classes)
        metrics["client_id"] = client_id
        metrics["skipped"] = False
        return metrics

    def aggregate_metrics(
        self,
        client_metrics: List[Dict],
        round_num: int = 0,
    ) -> Dict:
        """
        Weighted average of client metrics by num_samples.
        Reports global estimate + per-client breakdown.
        """
        valid = [m for m in client_metrics if not m.get("skipped", False)]
        if not valid:
            return {"round": round_num, "error": "No valid client evaluations"}

        total_samples = sum(m["num_samples"] for m in valid)
        weights = [m["num_samples"] / total_samples for m in valid]

        global_metrics = {
            "round": round_num,
            "num_clients": len(valid),
            "total_eval_samples": total_samples,
            "accuracy": float(sum(w * m["accuracy"] for w, m in zip(weights, valid))),
            "f1_macro": float(sum(w * m["f1_macro"] for w, m in zip(weights, valid))),
            "f1_weighted": float(sum(w * m["f1_weighted"] for w, m in zip(weights, valid))),
            "auc_roc": float(sum(w * m["auc_roc"] for w, m in zip(weights, valid))),
            "loss": float(sum(w * m["loss"] for w, m in zip(weights, valid))),
            "per_client": {m["client_id"]: m for m in valid},
        }

        # Detect heterogeneity: std across clients
        accs = [m["accuracy"] for m in valid]
        global_metrics["accuracy_std"] = float(np.std(accs))
        global_metrics["worst_client_accuracy"] = float(min(accs))
        global_metrics["best_client_accuracy"] = float(max(accs))

        self._eval_history.append(global_metrics)

        logger.info(
            f"[FedEval Round {round_num}] "
            f"acc={global_metrics['accuracy']:.4f} ± {global_metrics['accuracy_std']:.4f} | "
            f"f1={global_metrics['f1_macro']:.4f} | "
            f"worst_client={global_metrics['worst_client_accuracy']:.4f}"
        )
        return global_metrics

    def run_evaluation_round(
        self,
        model: nn.Module,
        client_data: Dict[int, tuple],
        round_num: int = 0,
        batch_size: int = 256,
    ) -> Dict:
        """
        Full round: evaluate each client → aggregate.
        client_data: {client_id: (X_val, y_val)}
        """
        client_metrics = []
        for client_id, (X_val, y_val) in client_data.items():
            metrics = self.client_evaluate(client_id, model, X_val, y_val, batch_size)
            client_metrics.append(metrics)

        return self.aggregate_metrics(client_metrics, round_num)

    def get_eval_history(self) -> List[Dict]:
        return self._eval_history

    def detect_performance_drift(
        self,
        window: int = 5,
        threshold: float = 0.03,
    ) -> Dict:
        """Check if accuracy dropped significantly in recent rounds."""
        if len(self._eval_history) < window + 1:
            return {"drift_detected": False, "reason": "Not enough history"}

        recent = self._eval_history[-window:]
        older = self._eval_history[-(2 * window):-window]

        recent_avg = np.mean([r["accuracy"] for r in recent])
        older_avg = np.mean([r["accuracy"] for r in older])
        drop = older_avg - recent_avg

        return {
            "drift_detected": drop > threshold,
            "accuracy_drop": float(drop),
            "recent_avg_accuracy": float(recent_avg),
            "older_avg_accuracy": float(older_avg),
            "threshold": threshold,
        }
