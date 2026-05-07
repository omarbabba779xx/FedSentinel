import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Any


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def flatten_params(model: torch.nn.Module) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_norm(params: List[np.ndarray]) -> float:
    flat = np.concatenate([p.flatten() for p in params])
    return float(np.linalg.norm(flat))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def weights_to_numpy(weights) -> List[np.ndarray]:
    return [w if isinstance(w, np.ndarray) else np.array(w) for w in weights]


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in keys}
