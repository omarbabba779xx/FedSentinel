"""
IID and Non-IID data splitting for federated learning simulation.
Non-IID via Dirichlet distribution — each client gets skewed class distribution.
"""

import numpy as np
from typing import List, Tuple, Dict
from utils.logger import get_logger

logger = get_logger("DataSplitter")


def iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, num_clients)
    result = [(X[idx], y[idx]) for idx in splits]
    _log_split_stats("IID", result)
    return result


def non_iid_dirichlet_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Dirichlet-based non-IID split.
    alpha → 0 : extreme heterogeneity (each client sees ~1 class)
    alpha → ∞ : approaches IID
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    num_classes = len(classes)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(ci) < len(X) / num_clients) for p, ci in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()

        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[i].extend(chunk.tolist())

    result = []
    for idx in client_indices:
        idx = np.array(idx)
        if len(idx) < min_samples_per_client:
            extra = rng.choice(len(X), min_samples_per_client - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))

    _log_split_stats(f"Non-IID Dirichlet (α={alpha})", result)
    return result


def pathological_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each client gets exactly `classes_per_client` classes."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    num_classes = len(classes)

    class_assignments = []
    all_class_combos = []
    for i in range(num_clients):
        assigned = classes[(np.arange(classes_per_client) + i * classes_per_client) % num_classes]
        all_class_combos.append(assigned)

    result = []
    for assigned_classes in all_class_combos:
        mask = np.isin(y, assigned_classes)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = rng.choice(len(X), 100)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))

    _log_split_stats(f"Pathological ({classes_per_client} classes/client)", result)
    return result


def get_client_stats(client_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    stats = []
    for i, (X, y) in enumerate(client_data):
        unique, counts = np.unique(y, return_counts=True)
        stats.append({
            "client_id": i,
            "total_samples": len(y),
            "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
        })
    return stats


def _log_split_stats(split_type: str, client_data: List[Tuple[np.ndarray, np.ndarray]]):
    logger.info(f"[{split_type}] {len(client_data)} clients:")
    for i, (X, y) in enumerate(client_data):
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"  Client {i}: {len(y)} samples | classes: {dist}")
