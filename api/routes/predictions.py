import time
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Depends
from api.schemas.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from utils.logger import get_logger

router = APIRouter(prefix="/predict", tags=["Predictions"])
logger = get_logger("PredictionAPI")

ATTACK_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def get_model_state():
    from api.main import app
    return app.state


@router.post("/single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, state=Depends(get_model_state)):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Start FL training first.")

    t0 = time.perf_counter()
    x = np.array(request.features, dtype=np.float32)

    if state.preprocessor is not None:
        expected = state.preprocessor.num_features
        if len(x) != expected:
            raise HTTPException(status_code=422, detail=f"Expected {expected} features, got {len(x)}")

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(state.device)

    state.model.eval()
    with torch.no_grad():
        out = state.model(x_tensor)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        import torch.nn.functional as F
        proba = F.softmax(logits, dim=-1).cpu().numpy()[0]

    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    latency = (time.perf_counter() - t0) * 1000

    explanation = None
    if request.explain and state.shap_explainer is not None:
        try:
            explanation = state.shap_explainer.explain_single(x)
        except Exception as e:
            logger.warning(f"Explanation failed: {e}")

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_class_name=ATTACK_NAMES[predicted_class] if predicted_class < len(ATTACK_NAMES) else str(predicted_class),
        confidence=confidence,
        probabilities={ATTACK_NAMES[i]: float(proba[i]) for i in range(len(proba))},
        explanation=explanation,
        model_round=state.current_round,
        latency_ms=round(latency, 3),
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, state=Depends(get_model_state)):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    X = np.array(request.samples, dtype=np.float32)
    x_tensor = torch.tensor(X, dtype=torch.float32).to(state.device)

    state.model.eval()
    with torch.no_grad():
        out = state.model(x_tensor)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        import torch.nn.functional as F
        proba = F.softmax(logits, dim=-1).cpu().numpy()

    predictions = []
    for i in range(len(X)):
        pred_cls = int(np.argmax(proba[i]))
        predictions.append(PredictionResponse(
            predicted_class=pred_cls,
            predicted_class_name=ATTACK_NAMES[pred_cls] if pred_cls < len(ATTACK_NAMES) else str(pred_cls),
            confidence=float(proba[i][pred_cls]),
            probabilities={ATTACK_NAMES[j]: float(proba[i][j]) for j in range(len(proba[i]))},
            model_round=state.current_round,
            latency_ms=0,
        ))

    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        predictions=predictions,
        total_samples=len(X),
        processing_time_ms=round(total_ms, 3),
    )
