# api/main.py
import os
import time
# import json
import logging
import warnings
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.io_utils import (
    load_json,
    load_model,
    _mlflow_available,
    load_model_from_mlflow,
    load_artifact_json_from_mlflow,
)
from src.drift import load_baseline, detect_drift
from api.schemas import PredictRequest, PredictResponse, DriftRequest, DriftResponse

# Suppress Pydantic protected namespace warning from MLflow internals
warnings.filterwarnings("ignore", message="Field \"model_.*\" has conflict with protected namespace")


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
DRIFT_PSI_AGGREGATE = Gauge("drift_psi_aggregate", "Aggregate PSI score from last drift check")
DRIFT_DETECTED = Gauge("drift_detected", "1 if drift was detected, 0 otherwise")
DRIFT_FEATURES_COUNT = Gauge("drift_features_drifted", "Number of features with detected drift")



# ---------- App + model loading ----------
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Configuration — dual-mode: MLflow (default) or local file fallback
MODEL_PATH = os.getenv("MODEL_PATH", "")  # If set, forces local mode
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-model")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

model = None
feature_cols = None
_load_source = "unknown"   # Track where the model was loaded from
_baseline = None            # Drift detection baseline
_last_drift_report = None   # Cache last drift report


@app.on_event("startup")
def load_artifacts():
    global model, feature_cols, MODEL_VERSION, _load_source

    print("Starting up, loading model and schema...")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}"
          f"Local model path: {MODEL_PATH}"
          f"Available schema path: {_mlflow_available()}"
          f"Local schema path: {SCHEMA_PATH}"
          )
    # ---- Strategy 1: MLflow Model Registry ----
    if MLFLOW_TRACKING_URI and not MODEL_PATH and _mlflow_available():
        print("Attempting to load model from MLflow...")
        try:
            logger.info("Attempting MLflow model load from %s", MLFLOW_TRACKING_URI)

            print(f"Model loaded from MLflow: {MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_STAGE}")

            model = load_model_from_mlflow(
                tracking_uri=MLFLOW_TRACKING_URI,
                model_name=MLFLOW_MODEL_NAME,
                stage_or_version=MLFLOW_MODEL_STAGE,
            )

            print(f"Model loaded from MLflow: {MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_STAGE}")
            print("model:", model)

            # Try to load schema from MLflow artifacts
            try:
                schema = load_artifact_json_from_mlflow(
                    tracking_uri=MLFLOW_TRACKING_URI,
                    model_name=MLFLOW_MODEL_NAME,
                    artifact_path="schema.json",
                    stage_or_version=MLFLOW_MODEL_STAGE,
                )
                feature_cols = schema["features"]
            except Exception:
                # Fallback: derive features from model input schema if available
                logger.warning("Could not load schema.json from MLflow, using default feature list")
                feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

            # Resolve registry version for reporting
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                client = MlflowClient()
                
                if MLFLOW_MODEL_STAGE.isdigit():
                    mv = client.get_model_version(MLFLOW_MODEL_NAME, int(MLFLOW_MODEL_STAGE))
                else:
                    # On utilise l'Alias au lieu du Stage
                    try:
                        mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
                    except:
                        # Fallback si l'alias n'est pas encore utilisé
                        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE])
                        mv = versions[0] if versions else None
                        
                if mv:
                    MODEL_VERSION = f"mlflow-v{mv.version}"
            except Exception as e:
                logger.warning("Could not resolve model version detail: %s", e)
                MODEL_VERSION = f"mlflow-{MLFLOW_MODEL_STAGE}"

            _load_source = f"mlflow:{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_STAGE}"

            logger.info(
                "model_loaded_from_mlflow",
                extra={
                    "model_name": MLFLOW_MODEL_NAME,
                    "stage": MLFLOW_MODEL_STAGE,
                    "model_version": MODEL_VERSION,
                    "threshold": THRESHOLD,
                },
            )
            return  # Success — skip local fallback
        except Exception as e:
            logger.warning("MLflow load failed, falling back to local: %s", e)

    # ---- Strategy 2: Local file (fallback) ----
    local_model_path = MODEL_PATH or "models/model_v1.joblib"
    local_schema_path = SCHEMA_PATH or "models/model_v1_schema.json"

    model = load_model(local_model_path)
    schema = load_json(local_schema_path)
    feature_cols = schema["features"]
    _load_source = f"local:{local_model_path}"

    logger.info(
        "model_loaded_from_local",
        extra={
            "model_path": local_model_path,
            "schema_path": local_schema_path,
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
        "load_source": _load_source,
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


@app.post("/drift", response_model=DriftResponse)
def drift_check(payload: DriftRequest):
    """Run drift detection on a batch of transactions."""
    global _baseline, _last_drift_report

    # Load baseline on first call (or if not yet loaded)
    if _baseline is None:
        try:
            _baseline = load_baseline(
                local_path=os.getenv("DRIFT_BASELINE_PATH", "models/model_v1_baseline.json"),
                mlflow_tracking_uri=MLFLOW_TRACKING_URI or None,
                mlflow_model_name=MLFLOW_MODEL_NAME or None,
                mlflow_stage=MLFLOW_MODEL_STAGE,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"Baseline not found: {e}")

    live_df = pd.DataFrame(payload.data)

    psi_threshold = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25"))
    z_threshold = float(os.getenv("DRIFT_Z_THRESHOLD", "2.0"))

    report = detect_drift(
        live_df=live_df,
        baseline=_baseline,
        features=feature_cols,
        psi_threshold=psi_threshold,
        z_threshold=z_threshold,
    )

    # Update Prometheus gauges
    DRIFT_PSI_AGGREGATE.set(report["aggregate_psi"])
    DRIFT_DETECTED.set(1.0 if report["drift_detected"] else 0.0)
    DRIFT_FEATURES_COUNT.set(
        len(set(report["features_drifted_psi"]) | set(report["features_drifted_mean"]))
    )

    _last_drift_report = report

    logger.info(
        "drift_check",
        extra={
            "drift_detected": report["drift_detected"],
            "aggregate_psi": report["aggregate_psi"],
            "n_features_drifted": len(report["features_drifted_psi"]),
            "n_samples": report["n_samples"],
        },
    )

    return DriftResponse(**report)


@app.get("/drift/latest")
def drift_latest():
    """Return the last drift report (or 404 if none yet)."""
    if _last_drift_report is None:
        raise HTTPException(status_code=404, detail="No drift check has been run yet.")
    return _last_drift_report
