"""
Comprehensive evaluation metrics for IDS model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score,
)
from utils.logger import get_logger

logger = get_logger("Metrics")

ATTACK_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Dict:
    model.eval()
    all_preds, all_targets, all_proba = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            import torch.nn.functional as F
            proba = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(proba, axis=-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            all_proba.append(proba)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_proba = np.vstack(all_proba)

    return compute_metrics(y_true, y_pred, y_proba, num_classes)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: int = 5,
) -> Dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=ATTACK_NAMES[:num_classes],
            zero_division=0,
        ),
    }

    if y_proba is not None:
        try:
            if num_classes == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc_roc"] = float(roc_auc_score(
                    y_true, y_proba,
                    multi_class="ovr", average="macro",
                    labels=list(range(num_classes)),
                ))
        except ValueError as e:
            logger.warning(f"AUC-ROC computation failed: {e}")
            metrics["auc_roc"] = 0.0

        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            metrics["average_precision"] = float(average_precision_score(y_bin, y_proba, average="macro"))
        except Exception:
            metrics["average_precision"] = 0.0

    # Detection rate per class
    cm = np.array(metrics["confusion_matrix"])
    dr_per_class = []
    for i in range(num_classes):
        tp = cm[i, i] if i < len(cm) else 0
        total = cm[i].sum() if i < len(cm) else 1
        dr_per_class.append(float(tp / total) if total > 0 else 0.0)
    metrics["detection_rate_per_class"] = dr_per_class

    # False positive rate
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - (fp + fn + np.diag(cm))
    fpr = fp / (fp + tn + 1e-8)
    metrics["false_positive_rate"] = float(fpr.mean())
    metrics["fpr_per_class"] = fpr.tolist()

    return metrics


def print_metrics_table(metrics: Dict, title: str = "Evaluation Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:         {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 (Macro):       {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 (Weighted):    {metrics.get('f1_weighted', 0):.4f}")
    print(f"  AUC-ROC:          {metrics.get('auc_roc', 0):.4f}")
    print(f"  Precision:        {metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall:           {metrics.get('recall_macro', 0):.4f}")
    print(f"  FPR:              {metrics.get('false_positive_rate', 0):.4f}")
    print(f"{'='*60}")
    f1_per = metrics.get("f1_per_class", [])
    for i, f1 in enumerate(f1_per):
        name = ATTACK_NAMES[i] if i < len(ATTACK_NAMES) else f"class_{i}"
        print(f"  F1 [{name:12s}]:  {f1:.4f}")
    print(f"{'='*60}\n")
    if "classification_report" in metrics:
        print(metrics["classification_report"])
