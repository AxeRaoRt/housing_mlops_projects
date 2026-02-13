import time
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST




from fastapi import FastAPI
from pathlib import Path
import os
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="California Housing API")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/v1/model.joblib"))
MODEL = None

PRED_REQUESTS = Counter("prediction_requests_total", "Total number of prediction requests")
PRED_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")


FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.on_event("startup")
def load_model():
    global MODEL
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
    MODEL = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(inp: HousingInput):
    if MODEL is None:
        return {"error": "model not loaded"}

    t0 = time.time()

    df = pd.DataFrame([inp.model_dump()], columns=FEATURES)
    pred = float(MODEL.predict(df)[0])

    latency = time.time() - t0

    PRED_REQUESTS.inc()
    PRED_LATENCY.observe(latency)

    return {
        "prediction": pred,
        "latency_seconds": latency
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
