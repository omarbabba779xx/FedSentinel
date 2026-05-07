"""
Krum and Multi-Krum aggregation (Blanchard et al., 2017).
Select update(s) with minimum sum of distances to nearest neighbors.
Provably Byzantine-robust when f < n/2.
"""

import numpy as np
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger("Krum")


def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])


def krum(
    client_weights: List[List[np.ndarray]],
    num_byzantine: int,
    return_scores: bool = False,
) -> Tuple[List[np.ndarray], dict]:
    """
    Krum: select the single update with minimum score.
    Score(i) = sum of squared distances to the (n-f-2) nearest neighbors.

    Args:
        client_weights: list of weight arrays per client
        num_byzantine:  f, expected number of Byzantine clients
        return_scores:  include per-client scores in returned dict

    Returns:
        aggregated weights, info dict
    """
    n = len(client_weights)
    f = num_byzantine
    k = n - f - 2  # neighbors to consider

    if k <= 0:
        logger.warning(f"Krum: k={k} ≤ 0 with n={n}, f={f}. Falling back to FedAvg.")
        avg = [np.mean([cw[i] for cw in client_weights], axis=0) for i in range(len(client_weights[0]))]
        return avg, {"selected": list(range(n)), "scores": []}

    flat = np.stack([_flatten_weights(cw) for cw in client_weights])  # (n, d)

    # Pairwise squared distances
    dists = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)  # (n, n)

    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        scores[i] = np.sum(sorted_dists[1:k + 1])  # exclude self

    selected_idx = int(np.argmin(scores))
    selected_weights = client_weights[selected_idx]

    info = {
        "selected": selected_idx,
        "scores": scores.tolist() if return_scores else [],
        "algorithm": "krum",
    }
    logger.info(f"Krum selected client {selected_idx} (score={scores[selected_idx]:.4f})")
    return selected_weights, info


def multi_krum(
    client_weights: List[List[np.ndarray]],
    num_byzantine: int,
    m: int = None,
) -> Tuple[List[np.ndarray], dict]:
    """
    Multi-Krum: select top-m clients by Krum score, then average.
    More robust than plain Krum; less biased than single-selection.
    """
    n = len(client_weights)
    f = num_byzantine
    if m is None:
        m = n - f

    k = n - f - 2
    if k <= 0:
        avg = [np.mean([cw[i] for cw in client_weights], axis=0) for i in range(len(client_weights[0]))]
        return avg, {"selected": list(range(n)), "algorithm": "multi_krum_fallback"}

    flat = np.stack([_flatten_weights(cw) for cw in client_weights])
    dists = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)

    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        scores[i] = np.sum(sorted_dists[1:k + 1])

    top_m_idx = np.argsort(scores)[:m]
    selected_clients = [client_weights[i] for i in top_m_idx]

    aggregated = [
        np.mean([cw[i] for cw in selected_clients], axis=0)
        for i in range(len(client_weights[0]))
    ]

    info = {"selected": top_m_idx.tolist(), "scores": scores.tolist(), "algorithm": "multi_krum"}
    logger.info(f"Multi-Krum selected clients {top_m_idx.tolist()} (m={m})")
    return aggregated, info
