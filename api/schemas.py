# api/schemas.py
from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
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
    model_version: str
    fraud_probability: float
    is_fraud: bool
