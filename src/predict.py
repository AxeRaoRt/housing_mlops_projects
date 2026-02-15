# src/predict.py
import argparse
import json
import os
import sys
from typing import Dict, Any, List

import pandas as pd

from src.io_utils import load_json, load_model


def build_features(payload: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    # Fail fast if keys missing
    missing = [c for c in feature_cols if c not in payload]
    extra = [k for k in payload.keys() if k not in feature_cols]

    if missing:
        raise ValueError(f"Missing required input fields: {missing}")
    # extra fields are not blocking, but we ignore them
    row = {k: float(payload[k]) for k in feature_cols}
    return pd.DataFrame([row], columns=feature_cols), extra


def main():
    parser = argparse.ArgumentParser(description="Offline inference for Fraud model")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models/model_v1.joblib"))
    parser.add_argument("--schema-path", default=os.getenv("SCHEMA_PATH", "models/model_v1_schema.json"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("PRED_THRESHOLD", "0.5")))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Inline JSON payload string")
    group.add_argument("--json-file", help="Path to JSON file payload")

    args = parser.parse_args()

    model = load_model(args.model_path)
    schema = load_json(args.schema_path)
    feature_cols = schema["features"]

    # Load payload
    if args.json:
        payload = json.loads(args.json)
    else:
        with open(args.json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

    X, extra = build_features(payload, feature_cols)
    proba = float(model.predict_proba(X)[:, 1][0])
    is_fraud = proba >= args.threshold

    output = {
        "model_path": args.model_path,
        "schema_path": args.schema_path,
        "threshold": args.threshold,
        "fraud_probability": proba,
        "is_fraud": bool(is_fraud),
        "extra_fields_ignored": extra,
    }

    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
