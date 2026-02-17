# api/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict

class PredictRequest(BaseModel):

    model_config = ConfigDict(protected_namespaces=())  # ← Résout le warning model_version

    # The dataset has these as required columns
    Time: float = Field(..., ge=0)
    Amount: float = Field(..., ge=0)

    # V1..V28
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # ← Résout le warning model_version

    model_version: str
    fraud_probability: float
    is_fraud: bool


class DriftRequest(BaseModel):
    """Batch of transactions for drift detection."""
    data: list[Dict[str, float]] = Field(..., min_length=2, description="List of transaction dicts")


class DriftFeatureReport(BaseModel):
    psi: float
    psi_drifted: bool
    actual_mean: float
    baseline_mean: float
    z_score: float
    mean_drifted: bool


class DriftResponse(BaseModel):
    n_samples: int
    n_features_checked: int
    features_drifted_psi: list[str]
    features_drifted_mean: list[str]
    per_feature: Dict[str, DriftFeatureReport]
    aggregate_psi: float
    drift_detected: bool
