"""
Robust aggregation rules:
- Trimmed Mean (Yin et al., 2018)
- Coordinate-wise Median
- FLAME (Nguyen et al., 2022) — clustering-based defense
- FLTrust (Cao et al., 2020) — server-side reference model
"""

import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("RobustAggregation")


def _flatten(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])


def _unflatten(flat: np.ndarray, reference: List[np.ndarray]) -> List[np.ndarray]:
    result = []
    idx = 0
    for ref in reference:
        size = ref.size
        result.append(flat[idx:idx + size].reshape(ref.shape))
        idx += size
    return result


# ──────────────────────────────────────────────────
# Trimmed Mean
# ──────────────────────────────────────────────────

def trimmed_mean(
    client_weights: List[List[np.ndarray]],
    beta: float = 0.1,
) -> Tuple[List[np.ndarray], dict]:
    """
    Coordinate-wise trimmed mean.
    Removes top and bottom beta fraction per coordinate before averaging.
    """
    n = len(client_weights)
    n_trim = max(1, int(np.floor(beta * n)))

    if 2 * n_trim >= n:
        logger.warning("Trimmed mean: too many clients trimmed, falling back to mean.")
        n_trim = max(0, n // 4)

    num_params = len(client_weights[0])
    aggregated = []

    for i in range(num_params):
        stacked = np.stack([cw[i] for cw in client_weights], axis=0)
        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[n_trim:n - n_trim]
        aggregated.append(np.mean(trimmed, axis=0))

    info = {"algorithm": "trimmed_mean", "beta": beta, "trimmed_per_side": n_trim, "remaining": n - 2 * n_trim}
    logger.info(f"Trimmed mean: β={beta}, trimmed {n_trim} clients per side")
    return aggregated, info


# ──────────────────────────────────────────────────
# Coordinate-wise Median
# ──────────────────────────────────────────────────

def coordinate_median(
    client_weights: List[List[np.ndarray]],
) -> Tuple[List[np.ndarray], dict]:
    aggregated = [
        np.median(np.stack([cw[i] for cw in client_weights], axis=0), axis=0)
        for i in range(len(client_weights[0]))
    ]
    info = {"algorithm": "coordinate_median"}
    return aggregated, info


# ──────────────────────────────────────────────────
# FLAME
# ──────────────────────────────────────────────────

def flame(
    client_weights: List[List[np.ndarray]],
    noise_sigma: float = 0.001,
    epsilon_cluster: float = 0.5,
) -> Tuple[List[np.ndarray], dict]:
    """
    FLAME: Filtering + Aggregation via Model Exclusion.
    1. Cluster updates with HDBSCAN (simulated with cosine distance threshold here)
    2. Keep only the majority cluster
    3. Add calibrated noise for DP

    Nguyen et al. "FLAME: Taming Backdoors in Federated Learning" (USENIX 2022).
    """
    try:
        from sklearn.cluster import DBSCAN
        USE_DBSCAN = True
    except ImportError:
        USE_DBSCAN = False

    n = len(client_weights)
    flat_updates = np.stack([_flatten(cw) for cw in client_weights])

    # Cosine similarity matrix
    norms = np.linalg.norm(flat_updates, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normalized = flat_updates / norms
    cos_sim = normalized @ normalized.T  # (n, n)
    cos_dist = 1.0 - cos_sim

    if USE_DBSCAN:
        clustering = DBSCAN(eps=epsilon_cluster, min_samples=max(2, n // 3), metric="precomputed")
        labels = clustering.fit_predict(np.clip(cos_dist, 0, 2))
    else:
        # Fallback: threshold-based selection
        mean_update = flat_updates.mean(axis=0)
        sims = [float(np.dot(flat_updates[i], mean_update) / (np.linalg.norm(flat_updates[i]) * np.linalg.norm(mean_update) + 1e-8)) for i in range(n)]
        labels = np.array([0 if s > 0.5 else -1 for s in sims])

    # Keep majority cluster
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        selected_idx = list(range(n))
    else:
        majority_label = unique[np.argmax(counts)]
        selected_idx = [i for i, l in enumerate(labels) if l == majority_label]

    selected_clients = [client_weights[i] for i in selected_idx]

    aggregated = [
        np.mean([cw[i] for cw in selected_clients], axis=0)
        for i in range(len(client_weights[0]))
    ]

    # Add calibrated DP noise
    if noise_sigma > 0:
        aggregated = [
            w + np.random.normal(0, noise_sigma, w.shape).astype(np.float32)
            for w in aggregated
        ]

    info = {
        "algorithm": "flame",
        "selected_clients": selected_idx,
        "rejected_clients": [i for i in range(n) if i not in selected_idx],
        "cluster_labels": labels.tolist(),
    }
    logger.info(f"FLAME: {len(selected_idx)}/{n} clients kept, {n - len(selected_idx)} rejected")
    return aggregated, info


# ──────────────────────────────────────────────────
# FLTrust
# ──────────────────────────────────────────────────

def fltrust(
    client_weights: List[List[np.ndarray]],
    server_weights: List[np.ndarray],
    global_weights: List[np.ndarray],
) -> Tuple[List[np.ndarray], dict]:
    """
    FLTrust: server computes its own update on a small clean dataset.
    Scales each client's update by cosine similarity with server update.

    Cao et al. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS 2022).
    """
    # Server gradient = server_weights - global_weights
    server_grad = [sw - gw for sw, gw in zip(server_weights, global_weights)]
    server_flat = _flatten(server_grad)
    server_norm = np.linalg.norm(server_flat)

    if server_norm < 1e-8:
        avg = [np.mean([cw[i] for cw in client_weights], axis=0) for i in range(len(global_weights))]
        return avg, {"algorithm": "fltrust_fallback"}

    trust_scores = []
    scaled_updates = []

    for cw in client_weights:
        client_grad = [cw_i - gw for cw_i, gw in zip(cw, global_weights)]
        client_flat = _flatten(client_grad)
        client_norm = np.linalg.norm(client_flat)

        if client_norm < 1e-8:
            trust_scores.append(0.0)
            scaled_updates.append([np.zeros_like(g) for g in client_grad])
            continue

        cos_sim = float(np.dot(client_flat, server_flat) / (client_norm * server_norm))
        trust_score = max(0.0, cos_sim)  # ReLU to eliminate negative scores
        trust_scores.append(trust_score)

        scale = trust_score * server_norm / client_norm
        scaled_updates.append([g * scale for g in client_grad])

    total_trust = sum(trust_scores)
    if total_trust < 1e-8:
        logger.warning("FLTrust: all trust scores zero, falling back to global weights.")
        return global_weights, {"algorithm": "fltrust_zero_trust"}

    aggregated = [
        global_weights[i] + sum(t * su[i] for t, su in zip(trust_scores, scaled_updates)) / total_trust
        for i in range(len(global_weights))
    ]

    info = {
        "algorithm": "fltrust",
        "trust_scores": trust_scores,
        "total_trust": total_trust,
    }
    logger.info(f"FLTrust: trust scores={[f'{t:.3f}' for t in trust_scores]}")
    return aggregated, info
