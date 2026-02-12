from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import joblib 
import numpy as np
import json
from datetime import datetime
import os
import asyncio

app = FastAPI(title="California Housing Price Prediction API")

# charger le modèle et le scaler sauvegardés
model = None
scaler = None

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "CaliforniaHousingRF")
SCALER_ARTIFACT_PATH = os.getenv("SCALER_ARTIFACT_PATH", "preprocess/scaler.pkl")

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        def _load_sync():
            mlflow.set_tracking_uri(TRACKING_URI)
            client = MlflowClient(tracking_uri=TRACKING_URI)

            # 1) Récupérer toutes les versions (triées desc en général, mais on trie au cas où)
            mvs = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if not mvs:
                raise RuntimeError(f"Aucune version trouvée pour '{REGISTERED_MODEL_NAME}'")

            # Trie par numéro de version (desc) pour prendre la plus récente AVEC run_id
            def _ver_int(mv):
                v = getattr(mv, "version", "0")
                try:
                    return int(v)
                except Exception:
                    return 0

            mvs = sorted(mvs, key=_ver_int, reverse=True)

            mv = next((mv for mv in mvs if getattr(mv, "run_id", None)), None)
            if not mv:
                raise RuntimeError(
                    f"Aucune version de '{REGISTERED_MODEL_NAME}' n'a de run_id "
                    f"(elles proviennent probablement de Logged Models)."
                )

            run_id = mv.run_id
            version = mv.version

            # 2) Charger le modèle depuis CETTE version (pas latest)
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # 3) Télécharger le scaler depuis les artefacts de CE run
            local_scaler_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=SCALER_ARTIFACT_PATH,
                tracking_uri=TRACKING_URI,
            )
            loaded_scaler = joblib.load(local_scaler_path)

            return loaded_model, loaded_scaler, model_uri, run_id, version

        model, scaler, model_uri, run_id, version = await asyncio.to_thread(_load_sync)

        print(f"✅ Model loaded from: {model_uri} (version={version})")
        print(f"✅ Scaler loaded from run {run_id} artifact: {SCALER_ARTIFACT_PATH}")

    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {"message": "Welcome to the California Housing Price Prediction API"}

@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    
    try: 
        # preparer les données
        data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
                          features.Population, features.AveOccup, features.Latitude, features.Longitude]])
        
        # pretraitement et prediction
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)

        # Logger la prédiction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features.dict(),
            'prediction': float(prediction[0])
        }
        
        with open('predictions.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return {
            "predicted_price": float(prediction[0]),
            "unit": "hundreds of thousands of dollars"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/health")
def health_check():
    return {"status" : "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}