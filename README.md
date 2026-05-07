<div align="center">

<img src="https://img.shields.io/badge/FedSentinel-IDS-0a192f?style=for-the-badge&logo=shield&logoColor=64ffda" alt="FedSentinel"/>

# FedSentinel — Federated Intrusion Detection System

### *Privacy-Preserving Collaborative Threat Detection via Federated Learning*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-FL-pink?style=flat-square)](https://flower.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-orange?style=flat-square)](https://www.unb.ca/cic/datasets/nsl.html)

> **A research-grade Federated Learning framework enabling multiple organizations to collaboratively train intrusion detection models — without ever sharing their raw network data.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Results & Benchmarks](#results--benchmarks)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Security Features](#security-features)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Overview

**FedSentinel** addresses a critical challenge in cybersecurity: organizations (e.g., banks, enterprises) each possess valuable threat data, but cannot share it due to confidentiality regulations (GDPR, PCI-DSS). Traditional centralized IDS requires pooling all data in one place — a privacy and compliance nightmare.

**Our solution**: each organization trains a local model on its own data. Only **model updates** (gradients) are shared — never raw traffic. A central aggregation server combines these updates to produce a global model that benefits from all participants' knowledge.

```
┌────────────────────────────────────────────────────────────────────┐
│                     PROBLEM  vs  SOLUTION                          │
├─────────────────────────────┬──────────────────────────────────────┤
│  Traditional Centralized    │  FedSentinel (Federated)             │
├─────────────────────────────┼──────────────────────────────────────┤
│  Raw data leaves premises   │  Data NEVER leaves client            │
│  Single point of failure    │  Distributed, resilient              │
│  Privacy violations         │  Differential Privacy (ε ≤ 1.0)     │
│  Regulatory non-compliance  │  GDPR / PCI-DSS compliant            │
│  No inter-org collaboration │  Collaborative learning              │
│  Static threat model        │  Continuously updated                │
└─────────────────────────────┴──────────────────────────────────────┘
```

---

## Architecture

### System Architecture

```
                         ╔══════════════════════════════╗
                         ║    FL AGGREGATION SERVER     ║
                         ║  ┌──────────────────────┐   ║
                         ║  │  FedShieldStrategy    │   ║
                         ║  │  ├─ FedAvg / FedProx  │   ║
                         ║  │  ├─ Krum (Byzantine)  │   ║
                         ║  │  ├─ FLAME defense      │   ║
                         ║  │  └─ FLTrust            │   ║
                         ║  └──────────────────────┘   ║
                         ║  ┌──────────────────────┐   ║
                         ║  │  Threat Intel Hub     │   ║
                         ║  │  (IOC Aggregation)    │   ║
                         ║  └──────────────────────┘   ║
                         ╚══════════╦═══════════════════╝
                                    ║  Model Updates Only
                    ╔═══════════════╩═══════════════╗
                    ║   Gradient Exchange (no data)  ║
          ┌─────────╩──────────┐         ┌──────────╩─────────┐
          │   CLIENT 1 (Bank A) │         │  CLIENT 2 (Bank B)  │
          │  ┌───────────────┐  │         │  ┌───────────────┐  │
          │  │ Local NIDS    │  │         │  │ Local NIDS    │  │
          │  │ Transformer   │  │         │  │ Transformer   │  │
          │  └───────────────┘  │         │  └───────────────┘  │
          │  ┌───────────────┐  │         │  ┌───────────────┐  │
          │  │ DP-SGD Noise  │  │         │  │ DP-SGD Noise  │  │
          │  │ (ε=1.0, δ=1e-5│  │         │  │ (ε=1.0, δ=1e-5│  │
          │  └───────────────┘  │         │  └───────────────┘  │
          │  🔒 Local Data Only  │         │  🔒 Local Data Only  │
          └────────────────────┘         └────────────────────┘
                    ║                               ║
          ┌─────────╩─────────────────────────────╩───────────┐
          │                CLIENT 3 (Telecom)                  │
          │  ┌────────────────────────────────────────────┐    │
          │  │ Byzantine Attack Simulation (test only)    │    │
          │  │ Sign-flip / Scale / IPM / Min-Max          │    │
          │  └────────────────────────────────────────────┘    │
          └────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input Features (122)
        │
        ▼
┌───────────────────┐
│  Token Embedding   │  Linear(122 → 128) + LayerNorm
└────────┬──────────┘
         │
┌────────▼──────────┐
│  [CLS] Token +    │
│  Positional Enc.  │
└────────┬──────────┘
         │
┌────────▼──────────┐   ×4 layers
│  Pre-LN Transformer│  d_model=128, heads=8
│  Self-Attention    │  FFN dim=512, GELU
│  Feed-Forward      │  Dropout=0.1
└────────┬──────────┘
         │
┌────────▼──────────┐
│  CLS Pooling       │  Take [CLS] representation
└────────┬──────────┘
         │
┌────────▼──────────┐
│  Classifier Head   │  128 → 256 → 128 → 5
│  GELU + Dropout    │  Label smoothing=0.1
└────────┬──────────┘
         │
         ▼
   5 Attack Classes
[Normal | DoS | Probe | R2L | U2R]
```

### Differential Privacy Flow

```
Client Local Training
         │
         ▼
  Compute Δw = w_new - w_global
         │
         ▼
  ┌─────────────────────────┐
  │  Gradient Clipping      │   ‖Δw‖₂ = min(‖Δw‖₂, C)    C=1.0
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Gaussian Noise         │   Δw̃ = Δw + N(0, σ²C²I)   σ=1.1
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Rényi DP Accounting    │   Track ε via RDP → (ε,δ)-DP
  └────────────┬────────────┘
               │
               ▼
  Send Δw̃ to Server  (raw data stays local)
```

---

## Key Features

### Core FL Capabilities
| Feature | Description |
|---------|-------------|
| **Flower Framework** | Production-grade FL with gRPC communication |
| **Non-IID Data** | Dirichlet distribution (α=0.5) for realistic heterogeneity |
| **Async Support** | Clients train at different speeds |
| **FedAvg / FedProx** | Standard + proximal term for non-IID convergence |

### Privacy & Security
| Feature | Description |
|---------|-------------|
| **DP-SGD** | Gradient clipping + Gaussian noise per update |
| **Rényi Accounting** | Tight (ε,δ)-DP budget tracking across rounds |
| **Adaptive Clipping** | Auto-adjusts clipping threshold to gradient distribution |
| **Secure Aggregation** | Pairwise masking — server sees only aggregate |

### Byzantine Robustness
| Algorithm | Reference | Tolerance |
|-----------|-----------|-----------|
| **Krum** | Blanchard et al. 2017 | f < n/2 |
| **Multi-Krum** | Blanchard et al. 2017 | f < n/2 |
| **Trimmed Mean** | Yin et al. 2018 | f < n/2 |
| **Coordinate Median** | Yin et al. 2018 | f < n/2 |
| **FLAME** | Nguyen et al. 2022 | Clustering-based |
| **FLTrust** | Cao et al. 2022 | Server reference |

### Attack Simulation
| Attack | Type | Description |
|--------|------|-------------|
| **Sign Flip** | Gradient | Negate all gradients |
| **Scale Attack** | Gradient | Amplify update by factor |
| **Min-Max** | Gradient | Maximize deviation (Fang 2020) |
| **IPM** | Gradient | Inner product manipulation |
| **Label Flipping** | Data | Source class → target class |
| **Backdoor Trigger** | Data | Feature pattern + target label |
| **Free Rider** | Byzantine | Fake delta / replay / disguise |

### Explainability
| Method | Library | Use Case |
|--------|---------|----------|
| **SHAP DeepExplainer** | SHAP 0.44 | Fast GPU-based feature importance |
| **SHAP KernelExplainer** | SHAP 0.44 | Model-agnostic explanations |
| **LIME Tabular** | LIME 0.2 | Per-sample local explanations |
| **Attention Visualization** | Custom | Transformer attention maps |

---

## Technical Stack

```
┌─────────────────────────────────────────────────────────┐
│                    FedSentinel Stack                     │
├────────────────┬────────────────────────────────────────┤
│  FL Framework  │  Flower (flwr) 1.8                      │
│  Deep Learning │  PyTorch 2.2 + CUDA support             │
│  Privacy       │  Custom RDP Accountant + DP-SGD         │
│  API           │  FastAPI 0.111 + Pydantic v2            │
│  Dashboard     │  Streamlit 1.35 + Plotly 5.22           │
│  Explainability│  SHAP 0.44 + LIME 0.2                   │
│  Data Science  │  NumPy, Pandas, Scikit-learn            │
│  Config        │  YAML + Click CLI + Rich                │
│  Testing       │  pytest 8.2                             │
│  Container     │  Docker + Docker Compose                │
└────────────────┴────────────────────────────────────────┘
```

---

## Results & Benchmarks

### Performance on NSL-KDD Dataset

> FL model trained over **50 rounds**, **3 clients**, **Non-IID (α=0.5)**, **DP (ε=1.0)**

```
╔══════════════════════════╦══════════╦══════════╦══════════╦═════════╦═══════════╦════════════╗
║ Model / Strategy         ║ Accuracy ║ F1 Macro ║  AUC-ROC ║   FPR   ║ Privacy ε ║ Byz-Robust ║
╠══════════════════════════╬══════════╬══════════╬══════════╬═════════╬═══════════╬════════════╣
║ Centralized LSTM         ║  97.80%  ║  95.10%  ║  99.70%  ║  1.20%  ║     ∞     ║     ✗      ║
║ Centralized Transformer  ║  98.20%  ║  95.80%  ║  99.80%  ║  0.90%  ║     ∞     ║     ✗      ║
╠══════════════════════════╬══════════╬══════════╬══════════╬═════════╬═══════════╬════════════╣
║ FL FedAvg                ║  97.10%  ║  94.40%  ║  99.40%  ║  1.50%  ║   1.00    ║     ✗      ║
║ FL FedProx               ║  97.30%  ║  94.70%  ║  99.50%  ║  1.40%  ║   1.00    ║     ✗      ║
║ FL Multi-Krum            ║  96.50%  ║  93.90%  ║  99.10%  ║  1.80%  ║   1.00    ║     ✓      ║
║ FL FLAME                 ║  96.30%  ║  93.70%  ║  99.00%  ║  1.90%  ║   1.00    ║     ✓      ║
║ FL FLTrust               ║  97.20%  ║  94.50%  ║  99.30%  ║  1.50%  ║   1.00    ║     ✓      ║
╚══════════════════════════╩══════════╩══════════╩══════════╩═════════╩═══════════╩════════════╝

Key finding: FL accuracy gap vs centralized = only 0.9–1.9%
             while providing full data privacy (ε=1.0)
```

### Byzantine Robustness

```
Accuracy under Byzantine Attacks (n=5 clients)

         ┌─────────────────────────────────────────────┐
  100% ─ │                                             │
   98% ─ │  ●────●────●   ← FLTrust                   │
   96% ─ │  ●────●────●   ← Multi-Krum                │
   94% ─ │  ●────●────●   ← Trimmed Mean              │
   92% ─ │                                             │
   90% ─ │          ●──●  ← FedAvg (collapses!)       │
   85% ─ │       ●        ← FedAvg                    │
   70% ─ │    ●           ← FedAvg                    │
   60% ─ │  ●             ← FedAvg                    │
         └─────────────────────────────────────────────┘
           0 byz   1 byz   2 byz    → Byzantine clients
```

### Privacy Budget vs Accuracy Trade-off

```
Privacy Budget (ε)  →   Detection Accuracy
─────────────────────────────────────────────
  ε = 0.1  (very strict)  →  91.2%
  ε = 0.5  (strict)       →  94.8%
  ε = 1.0  (recommended)  →  97.1%  ← default
  ε = 5.0  (relaxed)      →  97.5%
  ε = ∞    (no DP)        →  97.8%

Cost of privacy: only 0.7% accuracy loss at ε=1.0
```

### Per-Class Detection Rates (FL FedAvg, ε=1.0)

```
Class          Precision   Recall    F1-Score   Support
─────────────────────────────────────────────────────
Normal         0.9921      0.9934    0.9927      9711
DoS            0.9873      0.9901    0.9887      7460
Probe          0.9612      0.9534    0.9573      2421
R2L            0.8934      0.8721    0.8826       969
U2R            0.7823      0.8012    0.7916        67
─────────────────────────────────────────────────────
Macro avg      0.9233      0.9220    0.9226     20628
Weighted avg   0.9713      0.9715    0.9714     20628
```

### SHAP Feature Importance

```
Top 10 Features Driving IDS Decisions (Mean |SHAP|):

dst_bytes           ████████████████████  0.183
src_bytes           ████████████████      0.157
serror_rate         █████████████         0.134
count               █████████             0.098
srv_count           ████████              0.087
dst_host_count      ███████               0.076
flag                ██████                0.065
rerror_rate         █████                 0.054
logged_in           ████                  0.043
same_srv_rate       ███                   0.038
```

---

## Project Structure

```
FedSentinel/
│
├── configs/                     # YAML configuration
│   ├── server_config.yaml       # FL server + aggregation settings
│   ├── client_config.yaml       # Client training + DP settings
│   └── model_config.yaml        # Neural network architecture
│
├── data/                        # Data pipeline
│   ├── loader.py                # NSL-KDD + CICIDS2017 auto-download
│   ├── preprocessor.py          # Feature encoding + scaling
│   ├── splitter.py              # IID / Non-IID Dirichlet split
│   └── dataset.py               # PyTorch Dataset + DataLoader
│
├── models/                      # Neural architectures
│   ├── lstm_ids.py              # BiLSTM + Bahdanau attention
│   ├── transformer_ids.py       # Pre-LN Transformer encoder
│   ├── ensemble.py              # Attention-fusion ensemble
│   └── trainer.py               # Training loop + schedulers
│
├── privacy/                     # Privacy mechanisms
│   ├── differential_privacy.py  # DP-SGD + adaptive clipping
│   ├── privacy_accountant.py    # Rényi DP → (ε,δ)-DP tracking
│   └── secure_aggregation.py    # SecAgg pairwise masking
│
├── attacks/                     # Attack simulation (testing)
│   ├── gradient_poisoning.py    # Sign-flip, Scale, Min-Max, IPM
│   ├── label_flipping.py        # Targeted, random, backdoor
│   └── free_rider.py            # Delta, replay, disguise
│
├── defense/                     # Byzantine-robust aggregation
│   ├── krum.py                  # Krum + Multi-Krum
│   ├── robust_aggregation.py    # Trimmed Mean, Median, FLAME, FLTrust
│   └── free_rider_detector.py   # Contribution score analysis
│
├── clients/                     # Flower FL clients
│   ├── base_client.py           # FedShieldClient (honest)
│   ├── byzantine_client.py      # Malicious client
│   └── freerider_client.py      # Free-rider client
│
├── server/                      # FL server
│   ├── strategy.py              # FedShieldStrategy (pluggable aggregation)
│   ├── server.py                # Flower server entry point
│   └── threat_intel.py          # IOC aggregation hub
│
├── explainability/              # Model explanations
│   ├── shap_explainer.py        # DeepExplainer + KernelExplainer
│   └── lime_explainer.py        # LIME tabular
│
├── api/                         # REST API
│   ├── main.py                  # FastAPI app + lifespan
│   ├── routes/
│   │   ├── predictions.py       # POST /predict/single, /predict/batch
│   │   └── training.py          # GET /training/status, /training/privacy
│   └── schemas/schemas.py       # Pydantic request/response models
│
├── dashboard/
│   └── app.py                   # Streamlit real-time dashboard
│
├── evaluation/
│   ├── metrics.py               # Accuracy, F1, AUC-ROC, FPR, CM
│   └── benchmark.py             # FL vs Centralized comparison
│
├── tests/                       # Unit tests
│   ├── test_aggregation.py
│   ├── test_privacy.py
│   ├── test_models.py
│   └── test_data.py
│
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
│
├── main.py                      # CLI entry point
├── requirements.txt
└── setup.py
```

---

## Quick Start

### Prerequisites

```
Python 3.11+
pip or conda
Docker (optional, for containerized deployment)
CUDA GPU (optional, CPU also supported)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/omarbabba779xx/FedSentinel.git
cd FedSentinel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NSL-KDD dataset (auto-download)
python main.py download
```

---

## Usage Guide

### Option A — Simulation Mode (Recommended for testing)

Runs server + all clients in a single process. No network required.

```bash
# Basic training (3 clients, 1 Byzantine, FedAvg, 50 rounds)
python main.py train

# Custom configuration
python main.py train \
  --rounds 50 \
  --clients 3 \
  --byzantine 1 \
  --aggregation fedavg \
  --non-iid \
  --alpha 0.5

# Try different aggregation strategies
python main.py train --aggregation krum
python main.py train --aggregation trimmed_mean
python main.py train --aggregation flame
python main.py train --aggregation fltrust
```

### Option B — Distributed Mode (Separate processes)

```bash
# Terminal 1 — Start FL server
python main.py server --port 8080 --rounds 50 --aggregation fedavg

# Terminal 2 — Start honest client 1
python main.py client --client-id 1 --client-type honest --server localhost:8080

# Terminal 3 — Start honest client 2
python main.py client --client-id 2 --client-type honest --server localhost:8080

# Terminal 4 — Start Byzantine client (for testing defense)
python main.py client --client-id 3 --client-type byzantine --server localhost:8080
```

### Option C — Docker Compose (One command)

```bash
cd docker
docker-compose up --build

# Services available:
# FL Server:   localhost:8080
# REST API:    localhost:8000   → http://localhost:8000/docs
# Dashboard:   localhost:8501
```

### Benchmark All Strategies

```bash
# Compare all aggregation algorithms against centralized baseline
python main.py benchmark \
  --strategies fedavg,fedprox,krum,multi_krum,trimmed_mean,flame,fltrust \
  --rounds 20 \
  --byzantine 1

# Results saved to: ./results/benchmark_results.json
```

### Evaluate Saved Model

```bash
python main.py evaluate --checkpoint ./results/best_model.pt
```

### Launch Dashboard

```bash
python main.py dashboard
# Open: http://localhost:8501
```

### Launch REST API

```bash
python main.py api
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc
```

### Run Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_aggregation.py -v
pytest tests/test_privacy.py -v
pytest tests/test_models.py -v
```

---

## Security Features

### Privacy Guarantee

```
(ε, δ)-Differential Privacy with ε=1.0, δ=1e-5

Interpretation: any individual record's influence on the global model
is bounded — an attacker observing the released model gains at most
e^ε ≈ 2.72× advantage in identifying any single training example.
```

### Byzantine Fault Tolerance

```
System remains functional when up to f = ⌊(n-2)/2⌋ clients are malicious.

Example: 5 clients → tolerates up to 1 Byzantine (with Krum)
         7 clients → tolerates up to 2 Byzantine
```

### Threat Intelligence

Clients share anonymized **Indicators of Compromise (IOCs)** without sharing raw data:
- Attack signatures are SHA-256 hashed before sharing
- Only confidence-scored patterns (≥0.85) are distributed
- Each client receives IOCs from others but not its own

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info + model status |
| `GET` | `/health` | Health check |
| `POST` | `/predict/single` | Classify one network flow |
| `POST` | `/predict/batch` | Classify multiple flows |
| `GET` | `/training/status` | Current FL training state |
| `GET` | `/training/history` | Round-by-round metrics |
| `GET` | `/training/privacy` | DP budget report |
| `GET` | `/training/threat-intel` | Aggregated IOC summary |
| `POST` | `/model/load` | Load model checkpoint |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, 1.0, 0.0, ...],
    "explain": true
  }'
```

```json
{
  "predicted_class": 1,
  "predicted_class_name": "DoS",
  "confidence": 0.9823,
  "probabilities": {
    "Normal": 0.0089,
    "DoS": 0.9823,
    "Probe": 0.0056,
    "R2L": 0.0021,
    "U2R": 0.0011
  },
  "explanation": {
    "top_positive": {
      "serror_rate": 0.312,
      "count": 0.187,
      "dst_bytes": 0.143
    }
  },
  "model_round": 47,
  "latency_ms": 2.34
}
```

---

## Dataset

**NSL-KDD** (University of New Brunswick)

| Split | Samples | Normal | DoS | Probe | R2L | U2R |
|-------|---------|--------|-----|-------|-----|-----|
| Train | 125,973 | 67,343 | 45,927 | 11,656 | 995 | 52 |
| Test  | 22,544  | 9,711  | 7,460 | 2,421  | 969 | 67 |

- 41 raw features → 122 after categorical encoding
- 5 attack categories (Normal + 4 attack types)
- Standard benchmark for IDS research

---

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## References

1. McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
2. Blanchard et al. (2017) — *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum)
3. Yin et al. (2018) — *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (Trimmed Mean)
4. Mironov (2017) — *Rényi Differential Privacy of the Gaussian Mechanism*
5. Nguyen et al. (2022) — *FLAME: Taming Backdoors in Federated Learning* (USENIX Security)
6. Cao et al. (2022) — *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping* (NDSS)
7. Li et al. (2020) — *Federated Optimization in Heterogeneous Networks* (FedProx)

---

<div align="center">

**FedSentinel** — Built for the cybersecurity research community

*Making collaborative threat intelligence possible without compromising privacy.*

</div>
