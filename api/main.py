"""
FedSentinel REST API
FastAPI application exposing prediction, training metrics, and privacy endpoints.
"""

import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import predictions, training
from utils.logger import get_logger
from utils.helpers import get_device

logger = get_logger("FedSentinelAPI")


class AppState:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.device = get_device()
        self.current_round = 0
        self.feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.__class__ = AppState
    app.state.__init__ = AppState.__init__
    app.state.model = None
    app.state.preprocessor = None
    app.state.shap_explainer = None
    app.state.lime_explainer = None
    app.state.device = get_device()
    app.state.current_round = 0
    app.state.feature_names = None

    # Auto-load model if saved checkpoint exists
    checkpoint = Path("./results/best_model.pt")
    if checkpoint.exists():
        try:
            from models import build_model
            state_dict = torch.load(checkpoint, map_location=app.state.device)
            arch = state_dict.pop("__architecture__", "transformer")
            input_size = state_dict.pop("__input_size__", 122)
            num_classes = state_dict.pop("__num_classes__", 5)
            model = build_model(arch, input_size=input_size, num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.eval()
            app.state.model = model.to(app.state.device)
            logger.info(f"Loaded model checkpoint from {checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    logger.info(f"FedSentinel API started | device={app.state.device}")
    yield
    logger.info("FedSentinel API shutting down")


app = FastAPI(
    title="FedSentinel API",
    description="""
    **Federated Learning Intrusion Detection System**

    Privacy-preserving collaborative threat detection.

    ## Features
    - Real-time network traffic classification (5 classes)
    - SHAP / LIME explanations per prediction
    - FL training monitoring (rounds, accuracy, privacy budget)
    - Byzantine attack detection status
    - Threat intelligence aggregation
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(training.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "FedSentinel",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": app.state.model is not None,
        "device": str(app.state.device),
    }


@app.get("/health", tags=["Health"])
async def health():
    return JSONResponse({"status": "healthy", "model_ready": app.state.model is not None})


@app.post("/model/load", tags=["Model"])
async def load_model(checkpoint_path: str = "./results/best_model.pt"):
    try:
        from models import build_model
        state_dict = torch.load(checkpoint_path, map_location=app.state.device)
        arch = state_dict.pop("__architecture__", "transformer")
        input_size = state_dict.pop("__input_size__", 122)
        num_classes = state_dict.pop("__num_classes__", 5)
        model = build_model(arch, input_size=input_size, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        app.state.model = model.to(app.state.device)
        return {"status": "loaded", "architecture": arch, "input_size": input_size}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
