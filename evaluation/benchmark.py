"""
Benchmark suite: compare FL aggregation strategies against centralized baseline.
Generates comprehensive comparison table and plots.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Dict
from collections import OrderedDict

from models import build_model, IDSTrainer, build_optimizer, build_scheduler
from data import make_dataloaders, train_val_split, compute_class_weights
from defense import krum, multi_krum, trimmed_mean, coordinate_median, flame
from evaluation.metrics import evaluate_model, print_metrics_table
from utils.logger import get_logger
from utils.helpers import get_device, save_json, set_seed

logger = get_logger("Benchmark")


class FLBenchmark:
    """
    Simulates FL with various aggregation strategies and compares results.
    No actual Flower server — runs in-process for speed.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_clients: int = 3,
        num_classes: int = 5,
        seed: int = 42,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.seed = seed
        self.device = get_device()
        self.input_size = X_train.shape[1]
        set_seed(seed)

    def _make_model(self):
        return build_model("transformer", self.input_size, self.num_classes).to(self.device)

    def _client_split(self):
        from data import non_iid_dirichlet_split
        return non_iid_dirichlet_split(self.X_train, self.y_train, self.num_clients, alpha=0.5)

    def _train_client(self, model, X_c, y_c, epochs=3):
        X_tr, y_tr, X_v, y_v = train_val_split(X_c, y_c)
        train_loader, val_loader = make_dataloaders(X_tr, y_tr, X_v, y_v, batch_size=256)
        cw = compute_class_weights(y_tr, self.num_classes).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        optimizer = build_optimizer(model, {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4})
        trainer = IDSTrainer(model, optimizer, criterion, device=self.device)
        trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False)
        return [p.cpu().numpy().copy() for p in model.parameters()]

    def run_fl_round(
        self,
        global_params: List[np.ndarray],
        client_splits,
        num_byzantine: int = 0,
        attack_type: str = "sign_flip",
        aggregation: str = "fedavg",
    ) -> List[np.ndarray]:
        from attacks import GradientPoisoningAttack

        client_updates = []
        for i, (X_c, y_c) in enumerate(client_splits):
            model = self._make_model()
            keys = list(model.state_dict().keys())
            state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, global_params)})
            model.load_state_dict(state)
            params = self._train_client(model, X_c, y_c)

            if i < num_byzantine:
                attack = GradientPoisoningAttack(attack_type)
                params = attack.poison(params)

            client_updates.append(params)

        if aggregation == "fedavg":
            return [np.mean([u[i] for u in client_updates], axis=0) for i in range(len(global_params))]
        elif aggregation == "krum":
            result, _ = krum(client_updates, num_byzantine)
            return result
        elif aggregation == "multi_krum":
            result, _ = multi_krum(client_updates, num_byzantine)
            return result
        elif aggregation == "trimmed_mean":
            result, _ = trimmed_mean(client_updates, beta=0.1)
            return result
        elif aggregation == "median":
            result, _ = coordinate_median(client_updates)
            return result
        elif aggregation == "flame":
            result, _ = flame(client_updates)
            return result
        return [np.mean([u[i] for u in client_updates], axis=0) for i in range(len(global_params))]

    def run(
        self,
        strategies: List[str] = None,
        num_rounds: int = 10,
        num_byzantine: int = 1,
        attack_type: str = "sign_flip",
    ) -> Dict:
        if strategies is None:
            strategies = ["fedavg", "krum", "multi_krum", "trimmed_mean", "flame"]

        test_loader, _ = make_dataloaders(self.X_test, self.y_test, self.X_test[:100], self.y_test[:100], batch_size=512)
        client_splits = self._client_split()
        results = {}

        # Centralized baseline
        logger.info("Training centralized baseline...")
        central_model = self._make_model()
        X_tr, y_tr, X_v, y_v = train_val_split(self.X_train, self.y_train)
        train_loader, val_loader = make_dataloaders(X_tr, y_tr, X_v, y_v, batch_size=256)
        cw = compute_class_weights(y_tr, self.num_classes).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=cw)
        opt = build_optimizer(central_model, {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4})
        trainer = IDSTrainer(central_model, opt, criterion, device=self.device)
        trainer.fit(train_loader, val_loader, epochs=num_rounds, verbose=False)
        central_metrics = evaluate_model(central_model, test_loader, self.device, self.num_classes)
        results["centralized"] = central_metrics
        print_metrics_table(central_metrics, "Centralized Baseline")

        # FL strategies
        for strategy in strategies:
            logger.info(f"Benchmarking strategy: {strategy} | byzantine={num_byzantine}")
            global_model = self._make_model()
            global_params = [p.cpu().numpy().copy() for p in global_model.parameters()]

            for r in range(num_rounds):
                global_params = self.run_fl_round(
                    global_params, client_splits, num_byzantine, attack_type, strategy
                )

            keys = list(global_model.state_dict().keys())
            state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, global_params)})
            global_model.load_state_dict(state)
            fl_metrics = evaluate_model(global_model, test_loader, self.device, self.num_classes)
            results[strategy] = fl_metrics
            print_metrics_table(fl_metrics, f"FL - {strategy.upper()} (byzantine={num_byzantine})")

        save_json(results, "./results/benchmark_results.json")
        self._print_comparison_table(results)
        return results

    def _print_comparison_table(self, results: Dict):
        print("\n" + "="*80)
        print(f"{'Strategy':<20} {'Accuracy':>10} {'F1 Macro':>10} {'AUC-ROC':>10} {'FPR':>10}")
        print("="*80)
        for name, m in results.items():
            print(f"{name:<20} {m.get('accuracy', 0):>10.4f} {m.get('f1_macro', 0):>10.4f} {m.get('auc_roc', 0):>10.4f} {m.get('false_positive_rate', 0):>10.4f}")
        print("="*80 + "\n")
