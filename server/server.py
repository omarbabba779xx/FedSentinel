"""
FL server entry point.
Starts Flower server with FedShieldStrategy.
"""

import flwr as fl
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from models import build_model
from server.strategy import FedShieldStrategy
from utils.logger import get_logger
from utils.config import load_all_configs
from utils.helpers import get_device

logger = get_logger("FedShieldServer")


def create_strategy(
    configs: dict,
    initial_model: Optional[torch.nn.Module] = None,
    input_size: int = 122,
    num_classes: int = 5,
) -> FedShieldStrategy:
    server_cfg = configs["server"]
    agg_cfg = server_cfg.get("aggregation", {})
    dp_cfg = configs["client"].get("privacy", {})
    bi_cfg = server_cfg.get("byzantine_detection", {})

    initial_parameters = None
    if initial_model is None:
        model_arch = configs["client"].get("model", {}).get("architecture", "transformer")
        initial_model = build_model(model_arch, input_size=input_size, num_classes=num_classes)

    if initial_model is not None:
        initial_parameters = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for val in initial_model.state_dict().values()]
        )

    Path("./results").mkdir(parents=True, exist_ok=True)

    return FedShieldStrategy(
        aggregation=agg_cfg.get("strategy", "fedavg"),
        num_byzantine=agg_cfg.get("krum_num_byzantine", 1),
        trimmed_mean_beta=agg_cfg.get("trimmed_mean_beta", 0.1),
        flame_noise_sigma=agg_cfg.get("flame_noise_sigma", 0.001),
        fedprox_mu=agg_cfg.get("fedprox_mu", 0.01),
        min_fit_clients=server_cfg["server"].get("min_fit_clients", 2),
        min_evaluate_clients=server_cfg["server"].get("min_evaluate_clients", 2),
        min_available_clients=server_cfg["server"].get("min_available_clients", 2),
        fraction_fit=server_cfg["server"].get("fraction_fit", 1.0),
        fraction_evaluate=server_cfg["server"].get("fraction_evaluate", 1.0),
        dp_enabled=dp_cfg.get("enabled", True),
        target_epsilon=dp_cfg.get("epsilon", 1.0),
        target_delta=dp_cfg.get("delta", 1e-5),
        free_rider_detection=bi_cfg.get("enabled", True),
        results_path=server_cfg["logging"].get("metrics_path", "./results/server_metrics.json"),
        initial_parameters=initial_parameters,
    )


def run_server(
    configs: dict,
    host: str = "0.0.0.0",
    port: int = 8080,
    num_rounds: int = 50,
    initial_model: Optional[torch.nn.Module] = None,
    input_size: int = 122,
    num_classes: int = 5,
):
    strategy = create_strategy(configs, initial_model, input_size, num_classes)
    address = f"{host}:{port}"

    logger.info(f"Starting FedShield FL server on {address} | rounds={num_rounds}")
    logger.info(f"Aggregation: {configs['server']['aggregation']['strategy'].upper()}")

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
