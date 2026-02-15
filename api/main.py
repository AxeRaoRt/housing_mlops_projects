# api/main.py
import os
import time
import json
import logging
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.io_utils import load_json, load_model
from api.schemas import PredictRequest, PredictResponse


# ---------- Logging (structured JSON) ----------
logger = logging.getLogger("fraud-api")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]


# ---------- Prometheus metrics ----------
REQUESTS_TOTAL = Counter("requests_total", "Total requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions")
ERRORS_TOTAL = Counter("errors_total", "Total errors")
FRAUD_PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predicted fraud transactions"
)



# ---------- App + model loading ----------
app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v1.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/model_v1_schema.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))

model = None
feature_cols = None


@app.on_event("startup")
def load_artifacts():
    global model, feature_cols
    model = load_model(MODEL_PATH)
    schema = load_json(SCHEMA_PATH)
    feature_cols = schema["features"]

    logger.info(
        "model_loaded",
        extra={
            "model_path": MODEL_PATH,
            "schema_path": SCHEMA_PATH,
            "model_version": MODEL_VERSION,
            "threshold": THRESHOLD,
        },
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method
    start = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception as e:
        ERRORS_TOTAL.inc()
        logger.exception("unhandled_exception", extra={"endpoint": endpoint, "error": str(e)})
        raise
    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        # status may not exist if exception thrown before response creation
        try:
            REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
        except Exception:
            REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status="500").inc()


@app.get("/health")
def health():
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):

    row = payload.model_dump()
    X = pd.DataFrame([row], columns=feature_cols)

    proba = float(model.predict_proba(X)[:, 1][0])
    is_fraud = proba >= THRESHOLD

    # 👇 HERE
    PREDICTIONS_TOTAL.inc()

    if is_fraud:
        FRAUD_PREDICTIONS_TOTAL.inc()

    logger.info(
        "prediction",
        extra={
            "model_version": MODEL_VERSION,
            "fraud_probability": proba,
            "is_fraud": is_fraud,
        },
    )

    return PredictResponse(
        model_version=MODEL_VERSION,
        fraud_probability=proba,
        is_fraud=is_fraud,
    )
