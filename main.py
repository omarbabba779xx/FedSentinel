"""
FedShield-IDS — Main entry point.

Commands:
  train       Run full FL training pipeline (server + clients in-process simulation)
  server      Start Flower FL server (for distributed mode)
  client      Start a Flower FL client (for distributed mode)
  benchmark   Run aggregation strategy benchmark
  evaluate    Evaluate saved model on test set
  explain     Generate SHAP explanations for current model
  api         Start FastAPI REST API
  dashboard   Start Streamlit dashboard
  download    Download NSL-KDD dataset
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
def cli():
    """FedShield-IDS: Privacy-Preserving Federated Intrusion Detection System"""
    console.print(Panel.fit(
        "[bold blue]FedShield-IDS[/bold blue] v1.0\n"
        "Privacy-Preserving Collaborative Threat Detection",
        border_style="blue",
    ))


@cli.command()
@click.option("--rounds", default=50, help="Number of FL rounds")
@click.option("--clients", default=3, help="Number of clients")
@click.option("--byzantine", default=1, help="Number of Byzantine clients")
@click.option("--aggregation", default="fedavg", help="Aggregation strategy")
@click.option("--non-iid", is_flag=True, default=True, help="Non-IID data split")
@click.option("--alpha", default=0.5, help="Dirichlet alpha for non-IID")
@click.option("--dataset", default="nsl_kdd", help="Dataset (nsl_kdd | cicids2017)")
@click.option("--architecture", default="transformer", help="Model architecture")
@click.option("--seed", default=42)
def train(rounds, clients, byzantine, aggregation, non_iid, alpha, dataset, architecture, seed):
    """Run FL training simulation (in-process, no network required)."""
    from utils.helpers import set_seed, get_device, save_json
    from utils.config import load_all_configs
    from data import load_dataset, NSLKDDPreprocessor, non_iid_dirichlet_split, iid_split
    from data import make_dataloaders, train_val_split, compute_class_weights
    from models import build_model, IDSTrainer, build_optimizer
    from evaluation import FLBenchmark, evaluate_model, print_metrics_table
    from collections import OrderedDict

    set_seed(seed)
    device = get_device()
    console.print(f"[green]Device:[/green] {device}")
    console.print(f"[green]Aggregation:[/green] {aggregation} | Byzantine={byzantine}/{clients}")

    # Load and preprocess data
    console.print("\n[yellow]Loading dataset...[/yellow]")
    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)
    preprocessor.save("./results/preprocessor.pkl")

    input_size = X_train.shape[1]
    num_classes = 5
    console.print(f"[green]Data:[/green] train={len(X_train)}, test={len(X_test)}, features={input_size}")

    # Run benchmark with all strategies
    console.print("\n[yellow]Running FL benchmark...[/yellow]")
    benchmark = FLBenchmark(X_train, y_train, X_test, y_test,
                             num_clients=clients, num_classes=num_classes, seed=seed)

    results = benchmark.run(
        strategies=[aggregation],
        num_rounds=rounds,
        num_byzantine=byzantine,
        attack_type="sign_flip",
    )

    # Save best model
    best_model = build_model(architecture, input_size, num_classes)
    Path("./results").mkdir(parents=True, exist_ok=True)
    state = best_model.state_dict()
    state["__architecture__"] = architecture
    state["__input_size__"] = input_size
    state["__num_classes__"] = num_classes
    torch.save(state, "./results/best_model.pt")

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Results saved to ./results/")


@cli.command()
@click.option("--port", default=8080)
@click.option("--rounds", default=50)
@click.option("--aggregation", default="fedavg")
def server(port, rounds, aggregation):
    """Start FL server (distributed mode)."""
    from utils.config import load_all_configs
    configs = load_all_configs()
    configs["server"]["server"]["port"] = port
    configs["server"]["server"]["num_rounds"] = rounds
    configs["server"]["aggregation"]["strategy"] = aggregation

    from server.server import run_server
    run_server(configs, port=port, num_rounds=rounds)


@cli.command()
@click.option("--client-id", required=True, type=int)
@click.option("--server", "server_address", default="localhost:8080")
@click.option("--client-type", default="honest", type=click.Choice(["honest", "byzantine", "freerider"]))
@click.option("--dataset", default="nsl_kdd")
def client(client_id, server_address, client_type, dataset):
    """Start a FL client (distributed mode)."""
    import flwr as fl
    from utils.config import load_all_configs
    from data import load_dataset, NSLKDDPreprocessor, non_iid_dirichlet_split
    from data.dataset import train_val_split

    configs = load_all_configs()

    df_train, _ = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)

    splits = non_iid_dirichlet_split(X_train, y_train, num_clients=5, alpha=0.5)
    X_c, y_c = splits[client_id % len(splits)]
    X_tr, y_tr, X_v, y_v = train_val_split(X_c, y_c)

    kwargs = dict(
        client_id=client_id,
        X_train=X_tr, y_train=y_tr,
        X_val=X_v, y_val=y_v,
        model_config=configs["model"],
        client_config=configs["client"],
        num_classes=5,
    )

    if client_type == "honest":
        from clients import FedShieldClient
        fl_client = FedShieldClient(**kwargs)
    elif client_type == "byzantine":
        from clients import ByzantineClient
        fl_client = ByzantineClient(**kwargs)
    else:
        from clients import FreeRiderClient
        fl_client = FreeRiderClient(**kwargs)

    fl.client.start_client(server_address=server_address, client=fl_client.to_client())


@cli.command()
@click.option("--strategies", default="fedavg,krum,trimmed_mean,flame", help="Comma-separated list")
@click.option("--rounds", default=20)
@click.option("--byzantine", default=1)
@click.option("--dataset", default="nsl_kdd")
def benchmark(strategies, rounds, byzantine, dataset):
    """Run full benchmark comparing aggregation strategies."""
    from data import load_dataset, NSLKDDPreprocessor
    from evaluation import FLBenchmark

    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    bench = FLBenchmark(X_train, y_train, X_test, y_test)
    bench.run(
        strategies=strategies.split(","),
        num_rounds=rounds,
        num_byzantine=byzantine,
    )


@cli.command()
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--dataset", default="nsl_kdd")
def evaluate(checkpoint, dataset):
    """Evaluate saved model on test set."""
    from data import load_dataset, NSLKDDPreprocessor
    from data.dataset import IDSDataset
    from torch.utils.data import DataLoader
    from models import build_model
    from evaluation import evaluate_model, print_metrics_table
    from utils.helpers import get_device
    import torch

    device = get_device()
    state_dict = torch.load(checkpoint, map_location=device)
    arch = state_dict.pop("__architecture__", "transformer")
    input_size = state_dict.pop("__input_size__", 122)
    num_classes = state_dict.pop("__num_classes__", 5)

    model = build_model(arch, input_size, num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    _, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor.load("./results/preprocessor.pkl") \
        if Path("./results/preprocessor.pkl").exists() else NSLKDDPreprocessor()

    if not preprocessor.is_fitted:
        df_train, _ = load_dataset(dataset)
        preprocessor.fit_transform(df_train)

    X_test, y_test = preprocessor.transform(df_test)
    test_ds = IDSDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    metrics = evaluate_model(model, test_loader, device, num_classes)
    print_metrics_table(metrics, f"Evaluation Results — {arch.upper()}")


@cli.command()
def api():
    """Start FastAPI REST API server."""
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


@cli.command()
def dashboard():
    """Start Streamlit monitoring dashboard."""
    import subprocess
    subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.port=8501"])


@cli.command()
def download():
    """Download NSL-KDD dataset."""
    from data.loader import NSLKDDLoader
    loader = NSLKDDLoader()
    loader.download()
    console.print("[green]NSL-KDD dataset downloaded![/green]")


if __name__ == "__main__":
    cli()
