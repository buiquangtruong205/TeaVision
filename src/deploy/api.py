"""FastAPI application serving the exported TeaVision CNN model."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.inference import TeaVisionPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "cnn_zscore" / "model_artifact.json"
SYSTEM_DIR = PROJECT_ROOT / "system"
MAX_UPLOAD_BYTES = int(os.getenv("TEAVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

logger = logging.getLogger("teavision.api")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    normalizer: str
    image_size: int
    class_names: list[str]
    metrics: dict
    leaf_localization: str


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float
    leaf_bbox: tuple[int, int, int, int]
    leaf_detected: bool
    localization_confidence: float = Field(ge=0.0, le=1.0)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("TEAVISION_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _artifact_path() -> Path:
    return Path(os.getenv("TEAVISION_MODEL_ARTIFACT", str(DEFAULT_ARTIFACT_PATH)))


async def _read_image_upload(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")
    return image_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    classify_leaf_crop = os.getenv("TEAVISION_CLASSIFY_LEAF_CROP", "false").lower() == "true"
    app.state.predictor = TeaVisionPredictor(
        _artifact_path(),
        classify_leaf_crop=classify_leaf_crop,
    )
    logger.info("Loaded TeaVision model from %s", _artifact_path())
    yield


app = FastAPI(
    title="TeaVision API",
    description="Tea leaf disease classification using the exported CNN + z-score model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=hasattr(request.app.state, "predictor"))


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    predictor: TeaVisionPredictor = request.app.state.predictor
    artifact = predictor.artifact
    return ModelInfoResponse(
        model_version=predictor.model_version,
        architecture=artifact.get("architecture", "SimpleCNN"),
        normalizer=artifact.get("normalizer", "zscore"),
        image_size=predictor.image_size,
        class_names=predictor.class_names,
        metrics=artifact.get("metrics", {}),
        leaf_localization=(
            "foreground contour with full-image fallback; "
            f"classify_leaf_crop={predictor.classify_leaf_crop}"
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await _read_image_upload(file)

    predictor: TeaVisionPredictor = request.app.state.predictor
    started = time.perf_counter()
    try:
        prediction = predictor.predict_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": request.client.host if request.client else None,
        "class_id": prediction.class_id,
        "class_name": prediction.class_name,
        "confidence": prediction.confidence,
        "latency_ms": latency_ms,
        "model_version": predictor.model_version,
        "leaf_bbox": prediction.leaf_bbox,
        "leaf_detected": prediction.leaf_detected,
        "localization_confidence": prediction.localization_confidence,
    }
    logger.info("prediction %s", json.dumps(log_record, ensure_ascii=True))

    return PredictionResponse(
        class_id=prediction.class_id,
        class_name=prediction.class_name,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        model_version=predictor.model_version,
        latency_ms=latency_ms,
        leaf_bbox=prediction.leaf_bbox,
        leaf_detected=prediction.leaf_detected,
        localization_confidence=prediction.localization_confidence,
    )


@app.post(
    "/predict/annotated",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def predict_annotated(request: Request, file: UploadFile = File(...)) -> Response:
    """Return the uploaded image with the detected leaf box and prediction label."""

    image_bytes = await _read_image_upload(file)
    predictor: TeaVisionPredictor = request.app.state.predictor
    try:
        prediction, annotated_bytes = predictor.predict_annotated_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=annotated_bytes,
        media_type="image/jpeg",
        headers={
            "X-TeaVision-Class": prediction.class_name,
            "X-TeaVision-Confidence": f"{prediction.confidence:.6f}",
        },
    )


app.mount("/system", StaticFiles(directory=SYSTEM_DIR), name="system-assets")
app.mount("/", StaticFiles(directory=SYSTEM_DIR, html=True), name="system-demo")
