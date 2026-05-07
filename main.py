"""
FedShield-IDS — Main entry point.

Commands:
  train           Run full FL training pipeline
  server          Start Flower FL server (distributed mode)
  client          Start a Flower FL client (distributed mode)
  benchmark       Run aggregation strategy benchmark
  evaluate        Evaluate saved model on test set
  explain         Generate SHAP explanations
  api             Start FastAPI REST API
  dashboard       Start Streamlit dashboard
  download        Download NSL-KDD dataset
  async-train     Run asynchronous FL with FedBuff
  zero-day        Run zero-day detection on test data
  live-capture    Start live packet capture IDS
  watermark       Embed / verify model watermark
  privacy-report  Print current privacy budget
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """FedShield-IDS: Privacy-Preserving Federated Intrusion Detection System"""
    console.print(Panel.fit(
        "[bold blue]FedShield-IDS[/bold blue] v2.0\n"
        "Privacy-Preserving Collaborative Threat Detection\n"
        "[dim]FL • DP • ZKP • HE • Blockchain • Zero-Day • GNN • MAML[/dim]",
        border_style="blue",
    ))


# ─── Training ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--rounds", default=50, help="Number of FL rounds")
@click.option("--clients", default=3, help="Number of clients")
@click.option("--byzantine", default=1, help="Number of Byzantine clients")
@click.option("--aggregation", default="fedavg",
              type=click.Choice(["fedavg", "fedprox", "krum", "multi_krum",
                                 "trimmed_mean", "coordinate_median", "flame", "fltrust"]))
@click.option("--non-iid", is_flag=True, default=True)
@click.option("--alpha", default=0.5, help="Dirichlet alpha for non-IID split")
@click.option("--dataset", default="nsl_kdd", type=click.Choice(["nsl_kdd", "cicids2017"]))
@click.option("--architecture", default="transformer",
              type=click.Choice(["transformer", "bilstm", "ensemble"]))
@click.option("--seed", default=42)
@click.option("--dp/--no-dp", default=True, help="Enable differential privacy")
@click.option("--noise-mult", default=1.0, help="DP noise multiplier")
@click.option("--epsilon", default=1.0, help="DP epsilon target")
def train(rounds, clients, byzantine, aggregation, non_iid, alpha, dataset,
          architecture, seed, dp, noise_mult, epsilon):
    """Run FL training simulation (in-process, no network required)."""
    from utils.helpers import set_seed, get_device
    from utils.config import load_all_configs
    from data import load_dataset, NSLKDDPreprocessor, non_iid_dirichlet_split, iid_split
    from data import make_dataloaders, train_val_split, compute_class_weights
    from models import build_model, IDSTrainer
    from evaluation import FLBenchmark

    set_seed(seed)
    device = get_device()
    console.print(f"[green]Device:[/green] {device}")
    console.print(f"[green]Config:[/green] aggregation={aggregation} | "
                  f"byzantine={byzantine}/{clients} | DP={dp} (ε={epsilon})")

    console.print("\n[yellow]Loading dataset...[/yellow]")
    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)
    preprocessor.save("./results/preprocessor.pkl")

    console.print(f"[green]Data:[/green] train={len(X_train)}, "
                  f"test={len(X_test)}, features={X_train.shape[1]}")

    console.print("\n[yellow]Running FL benchmark...[/yellow]")
    bench = FLBenchmark(X_train, y_train, X_test, y_test,
                        num_clients=clients, num_classes=5, seed=seed)
    results = bench.run(
        strategies=[aggregation],
        num_rounds=rounds,
        num_byzantine=byzantine,
        attack_type="sign_flip",
    )

    Path("./results").mkdir(parents=True, exist_ok=True)
    best_model = build_model(architecture, X_train.shape[1], 5)
    state = best_model.state_dict()
    state["__architecture__"] = architecture
    state["__input_size__"] = X_train.shape[1]
    state["__num_classes__"] = 5
    torch.save(state, "./results/best_model.pt")
    console.print(f"\n[bold green]Done![/bold green] Results → ./results/")


@cli.command()
@click.option("--rounds", default=50)
@click.option("--clients", default=5)
@click.option("--buffer-size", default=3, help="FedBuff async buffer size")
@click.option("--dataset", default="nsl_kdd")
@click.option("--architecture", default="transformer")
@click.option("--seed", default=42)
def async_train(rounds, clients, buffer_size, dataset, architecture, seed):
    """Asynchronous FL training using FedBuff (no synchronization barrier)."""
    from utils.helpers import set_seed, get_device
    from data import load_dataset, NSLKDDPreprocessor, non_iid_dirichlet_split
    from data.dataset import train_val_split
    from models import build_model
    from async_fl import FedBuffServer, AsyncFLClient, run_async_fl

    set_seed(seed)
    device = get_device()
    console.print(f"[green]AsyncFL:[/green] FedBuff | clients={clients} | buffer={buffer_size}")

    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    input_size = X_train.shape[1]
    global_model = build_model(architecture, input_size, 5)
    global_model.to(device)

    splits = non_iid_dirichlet_split(X_train, y_train, num_clients=clients)
    client_data = {}
    for cid, (Xc, yc) in enumerate(splits):
        Xtr, ytr, Xv, yv = train_val_split(Xc, yc)
        client_data[cid] = (Xtr, ytr)

    run_async_fl(
        global_model=global_model,
        client_data=client_data,
        num_rounds=rounds,
        buffer_size=buffer_size,
        device=device,
    )
    console.print("[bold green]AsyncFL training complete![/bold green]")


# ─── Server / Client (distributed) ────────────────────────────────────────

@cli.command()
@click.option("--port", default=8080)
@click.option("--rounds", default=50)
@click.option("--aggregation", default="fedavg")
def server(port, rounds, aggregation):
    """Start Flower FL server (distributed mode)."""
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
@click.option("--client-type", default="honest",
              type=click.Choice(["honest", "byzantine", "freerider"]))
@click.option("--dataset", default="nsl_kdd")
def client(client_id, server_address, client_type, dataset):
    """Start a Flower FL client (distributed mode)."""
    import flwr as fl
    from utils.config import load_all_configs
    from data import load_dataset, NSLKDDPreprocessor, non_iid_dirichlet_split
    from data.dataset import train_val_split

    configs = load_all_configs()
    df_train, _ = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    splits = non_iid_dirichlet_split(X_train, y_train, num_clients=5)
    X_c, y_c = splits[client_id % len(splits)]
    X_tr, y_tr, X_v, y_v = train_val_split(X_c, y_c)

    kwargs = dict(
        client_id=client_id,
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        model_config=configs["model"], client_config=configs["client"], num_classes=5,
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


# ─── Evaluation ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--strategies", default="fedavg,krum,trimmed_mean,flame")
@click.option("--rounds", default=20)
@click.option("--byzantine", default=1)
@click.option("--dataset", default="nsl_kdd")
def benchmark(strategies, rounds, byzantine, dataset):
    """Compare all aggregation strategies with/without Byzantine clients."""
    from data import load_dataset, NSLKDDPreprocessor
    from evaluation import FLBenchmark

    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)
    bench = FLBenchmark(X_train, y_train, X_test, y_test)
    bench.run(strategies=strategies.split(","), num_rounds=rounds, num_byzantine=byzantine)


@cli.command()
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--dataset", default="nsl_kdd")
def evaluate(checkpoint, dataset):
    """Evaluate saved model checkpoint on test set."""
    from data import load_dataset, NSLKDDPreprocessor
    from data.dataset import IDSDataset
    from torch.utils.data import DataLoader
    from models import build_model
    from evaluation import evaluate_model, print_metrics_table
    from utils.helpers import get_device

    device = get_device()
    state_dict = torch.load(checkpoint, map_location=device)
    arch = state_dict.pop("__architecture__", "transformer")
    input_size = state_dict.pop("__input_size__", 122)
    num_classes = state_dict.pop("__num_classes__", 5)

    model = build_model(arch, input_size, num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    _, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    df_train, _ = load_dataset(dataset)
    preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    test_ds = IDSDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    metrics = evaluate_model(model, test_loader, device, num_classes)
    print_metrics_table(metrics, f"Evaluation — {arch.upper()}")


@cli.command()
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--num-samples", default=100)
def explain(checkpoint, num_samples):
    """Generate SHAP explanations for model predictions."""
    from data import load_dataset, NSLKDDPreprocessor
    from models import build_model
    from explainability.shap_explainer import FedShieldSHAPExplainer
    from utils.helpers import get_device

    device = get_device()
    state_dict = torch.load(checkpoint, map_location=device)
    arch = state_dict.pop("__architecture__", "transformer")
    input_size = state_dict.pop("__input_size__", 122)
    num_classes = state_dict.pop("__num_classes__", 5)

    model = build_model(arch, input_size, num_classes).to(device)
    model.load_state_dict(state_dict)

    _, df_test = load_dataset("nsl_kdd")
    preprocessor = NSLKDDPreprocessor()
    df_train, _ = load_dataset("nsl_kdd")
    X_bg, _ = preprocessor.fit_transform(df_train)
    X_test, _ = preprocessor.transform(df_test)

    explainer = FedShieldSHAPExplainer(model, X_bg[:200], device=device)
    shap_vals = explainer.explain_batch(X_test[:num_samples])
    explainer.plot_summary(shap_vals, X_test[:num_samples])
    console.print(f"[green]SHAP explanations generated for {num_samples} samples[/green]")


# ─── Zero-day detection ────────────────────────────────────────────────────

@cli.command()
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--dataset", default="nsl_kdd")
@click.option("--threshold-percentile", default=95.0)
def zero_day(checkpoint, dataset, threshold_percentile):
    """Run zero-day anomaly detection (VAE + Isolation Forest ensemble)."""
    from data import load_dataset, NSLKDDPreprocessor
    from zero_day import ZeroDayDetector
    from utils.helpers import get_device

    device = get_device()
    console.print("[yellow]Fitting zero-day detector on normal traffic...[/yellow]")

    df_train, df_test = load_dataset(dataset)
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]

    detector = ZeroDayDetector(
        input_dim=X_train.shape[1],
        device=device,
        threshold_percentile=threshold_percentile,
    )
    detector.fit(X_normal)

    scores, flags = detector.predict(X_test)
    detected = flags.sum()
    console.print(f"[bold]Zero-day detection:[/bold] {detected}/{len(X_test)} samples flagged "
                  f"({100*detected/len(X_test):.1f}%)")

    report = detector.get_report()
    console.print(f"VAE threshold: {report.get('vae_threshold', 'N/A'):.4f}")


# ─── Live capture ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--interface", default=None, help="Network interface (default: system default)")
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--duration", default=60, help="Capture duration in seconds")
def live_capture(interface, checkpoint, duration):
    """Start live packet capture IDS (requires Scapy + root privileges)."""
    import time
    from models import build_model
    from live_capture import LiveIDSCapture
    from utils.helpers import get_device

    device = get_device()
    state_dict = torch.load(checkpoint, map_location=device)
    arch = state_dict.pop("__architecture__", "transformer")
    input_size = state_dict.pop("__input_size__", 41)
    num_classes = state_dict.pop("__num_classes__", 5)

    model = build_model(arch, input_size, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    def predict_callback(features: np.ndarray, meta: dict):
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred = torch.argmax(logits, dim=-1).item()
        if pred != 0:
            console.print(
                f"[red]ALERT:[/red] {CLASS_NAMES[pred]} | "
                f"src={meta.get('src_ip','?')} → dst={meta.get('dst_ip','?')}"
            )

    capture = LiveIDSCapture(
        prediction_callback=predict_callback,
        interface=interface,
    )
    capture.start()
    console.print(f"[green]Live capture started for {duration}s...[/green]")
    time.sleep(duration)
    capture.stop()
    console.print(f"[bold]Capture stats:[/bold] {capture.stats}")


# ─── Watermarking ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--checkpoint", default="./results/best_model.pt")
@click.option("--action", default="verify",
              type=click.Choice(["embed", "verify"]))
@click.option("--key-path", default="./results/watermark_key")
@click.option("--owner-id", default="FedSentinel")
def watermark(checkpoint, action, key_path, owner_id):
    """Embed or verify model ownership watermark."""
    import torch.nn as nn
    from models import build_model
    from watermarking import WatermarkGenerator, WatermarkEmbedder, OwnershipVerifier
    from utils.helpers import get_device

    device = get_device()
    state_dict = torch.load(checkpoint, map_location=device)
    arch = state_dict.pop("__architecture__", "transformer")
    input_size = state_dict.pop("__input_size__", 122)
    num_classes = state_dict.pop("__num_classes__", 5)

    model = build_model(arch, input_size, num_classes).to(device)
    model.load_state_dict(state_dict)

    if action == "embed":
        key = WatermarkGenerator.random_feature_pattern(
            num_triggers=50, input_dim=input_size, owner_id=owner_id,
        )
        key.save(key_path)
        embedder = WatermarkEmbedder(key, device=device)
        result = embedder.embed(model, nn.CrossEntropyLoss(), round_num=0)
        torch.save(model.state_dict(), checkpoint.replace(".pt", "_watermarked.pt"))
        console.print(f"[green]Watermark embedded![/green] Trigger accuracy: {result['trigger_accuracy']:.3f}")
        console.print(f"Key saved → {key_path}_meta.json")
    else:
        from watermarking import WatermarkKey
        key = WatermarkKey.load(key_path)
        embedder = WatermarkEmbedder(key, device=device)
        verifier = OwnershipVerifier(embedder)
        report = verifier.verify_ownership(model)

        t = Table(title="Ownership Verification")
        t.add_column("Field"); t.add_column("Value")
        for k, v in report.items():
            t.add_row(str(k), str(v))
        console.print(t)


# ─── Privacy report ────────────────────────────────────────────────────────

@cli.command()
@click.option("--noise-mult", default=1.0)
@click.option("--sample-rate", default=0.1)
@click.option("--rounds", default=50)
@click.option("--delta", default=1e-5)
def privacy_report(noise_mult, sample_rate, rounds, delta):
    """Print projected privacy budget for given training config."""
    from privacy.privacy_accountant import PrivacyAccountant

    acct = PrivacyAccountant(target_epsilon=10.0, target_delta=delta)
    for r in range(rounds):
        acct.step(noise_mult, sample_rate, num_steps=1)

    report = acct.get_report()

    t = Table(title=f"Privacy Budget — {rounds} rounds")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Current ε", f"{report['current_epsilon']:.4f}")
    t.add_row("δ", f"{delta:.2e}")
    t.add_row("Remaining budget", f"{report['remaining_budget']:.4f}")
    t.add_row("Accounting method", "min(RDP, GDP)")
    t.add_row("Noise multiplier σ", str(noise_mult))
    t.add_row("Sampling rate q", str(sample_rate))
    console.print(t)


# ─── API / Dashboard ───────────────────────────────────────────────────────

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
    NSLKDDLoader().download()
    console.print("[green]NSL-KDD dataset downloaded![/green]")


if __name__ == "__main__":
    cli()
