from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class AttackType(str, Enum):
    normal = "Normal"
    dos = "DoS"
    probe = "Probe"
    r2l = "R2L"
    u2r = "U2R"


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Network flow feature vector (122 values for NSL-KDD)")
    explain: bool = Field(False, description="Include SHAP/LIME explanation")


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None
    model_round: int
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]]
    explain: bool = False


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float


class TrainingStatus(BaseModel):
    is_training: bool
    current_round: int
    total_rounds: int
    progress_pct: float
    current_accuracy: Optional[float] = None
    current_loss: Optional[float] = None
    current_epsilon: Optional[float] = None
    aggregation_strategy: str
    num_active_clients: int


class RoundMetrics(BaseModel):
    round: int
    avg_accuracy: float
    avg_loss: float
    epsilon: float
    num_clients: int
    aggregation: str
    rejected_clients: List[int] = []


class PrivacyReport(BaseModel):
    target_epsilon: float
    target_delta: float
    current_epsilon: float
    remaining_budget: float
    total_rounds: int
    budget_exceeded: bool


class ThreatSummary(BaseModel):
    total_iocs: int
    attack_distribution: Dict[str, int]
    top_threats: List[Dict[str, Any]]
    contributing_clients: List[int]
    last_updated: str


class ClientStatus(BaseModel):
    client_id: int
    is_active: bool
    client_type: str  # honest | byzantine | freerider
    last_contribution_round: int
    total_samples: int
    avg_accuracy: float
    is_suspected: bool = False


class BenchmarkRequest(BaseModel):
    strategies: List[str] = Field(default=["fedavg", "krum", "trimmed_mean", "flame"])
    num_rounds: int = Field(default=20, ge=1, le=100)
    num_byzantine: int = Field(default=1, ge=0)
    attack_type: str = "sign_flip"
