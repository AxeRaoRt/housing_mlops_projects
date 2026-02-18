import argparse
import json
import os
import sys
from typing import Dict, Any, List

import pandas as pd

from src.io_utils import (
    load_json,
    load_model,
    _mlflow_available,
    # load_model_from_mlflow,
    load_artifact_json_from_mlflow,
)
from src.config import Settings


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
    settings = Settings()

    parser = argparse.ArgumentParser(description="Offline inference for Fraud model")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", ""))
    parser.add_argument("--schema-path", default=os.getenv("SCHEMA_PATH", ""))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("PRED_THRESHOLD", "0.5")))

    # MLflow model URI (overrides --model-path when provided)
    parser.add_argument(
        "--mlflow-model-uri",
        default=os.getenv("MLFLOW_MODEL_URI", ""),
        help="MLflow model URI, e.g. 'models:/fraud-model/Production'. Overrides --model-path.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Inline JSON payload string")
    group.add_argument("--json-file", help="Path to JSON file payload")

    args = parser.parse_args()

    # ---- Load model + schema ----
    if args.mlflow_model_uri and _mlflow_available():
        # MLflow mode
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        print(f"Loading model from MLflow: {args.mlflow_model_uri}")
        model = mlflow.sklearn.load_model(args.mlflow_model_uri)

        # Try to get schema from MLflow artifacts
        try:
            # Parse model name and stage/version from URI  models:/name/stage
            parts = args.mlflow_model_uri.replace("models:/", "").split("/")
            model_name = parts[0]
            stage_or_version = parts[1] if len(parts) > 1 else "Production"
            schema = load_artifact_json_from_mlflow(
                tracking_uri=settings.mlflow_tracking_uri,
                model_name=model_name,
                artifact_path="schema.json",
                stage_or_version=stage_or_version,
            )
            feature_cols = schema["features"]
        except Exception:
            # Fallback: default feature list
            feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            print("  ⚠️  Could not load schema from MLflow, using default features")
    else:
        # Local mode
        model_path = args.model_path or "models/model_v1.joblib"
        schema_path = args.schema_path or "models/model_v1_schema.json"

        model = load_model(model_path)
        schema = load_json(schema_path)
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
        "source": args.mlflow_model_uri or (args.model_path or "models/model_v1.joblib"),
        "threshold": args.threshold,
        "fraud_probability": proba,
        "is_fraud": bool(is_fraud),
        "extra_fields_ignored": extra,
    }

    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
