import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.schemas.schemas import TrainingStatus, RoundMetrics, PrivacyReport, ThreatSummary

router = APIRouter(prefix="/training", tags=["Training"])

RESULTS_PATH = "./results/server_metrics.json"


def _load_history():
    p = Path(RESULTS_PATH)
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    history = _load_history()
    if not history:
        return TrainingStatus(
            is_training=False, current_round=0, total_rounds=50,
            progress_pct=0.0, aggregation_strategy="fedavg", num_active_clients=0
        )
    last = history[-1]
    return TrainingStatus(
        is_training=True,
        current_round=last.get("round", 0),
        total_rounds=50,
        progress_pct=last.get("round", 0) / 50 * 100,
        current_accuracy=last.get("avg_accuracy"),
        current_loss=last.get("avg_loss"),
        current_epsilon=last.get("epsilon"),
        aggregation_strategy=last.get("aggregation", "fedavg"),
        num_active_clients=last.get("num_clients", 0),
    )


@router.get("/history", response_model=list)
async def get_training_history(limit: int = 50):
    history = _load_history()
    return history[-limit:]


@router.get("/round/{round_num}", response_model=RoundMetrics)
async def get_round_metrics(round_num: int):
    history = _load_history()
    match = [r for r in history if r.get("round") == round_num]
    if not match:
        raise HTTPException(status_code=404, detail=f"Round {round_num} not found")
    r = match[0]
    return RoundMetrics(
        round=r["round"],
        avg_accuracy=r.get("avg_accuracy", 0),
        avg_loss=r.get("avg_loss", 0),
        epsilon=r.get("epsilon", 0),
        num_clients=r.get("num_clients", 0),
        aggregation=r.get("aggregation", "fedavg"),
        rejected_clients=r.get("rejected_free_riders", []),
    )


@router.get("/privacy", response_model=PrivacyReport)
async def get_privacy_report():
    try:
        p = Path("./results/privacy_report.json")
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return PrivacyReport(**data)
    except Exception:
        pass
    return PrivacyReport(
        target_epsilon=1.0, target_delta=1e-5,
        current_epsilon=0.0, remaining_budget=1.0,
        total_rounds=0, budget_exceeded=False
    )


@router.get("/threat-intel", response_model=ThreatSummary)
async def get_threat_intel():
    p = Path("./results/threat_intel.json")
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return ThreatSummary(**data)
    return ThreatSummary(
        total_iocs=0, attack_distribution={}, top_threats=[],
        contributing_clients=[], last_updated="N/A"
    )
