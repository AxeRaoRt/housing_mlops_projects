from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models") / "v1" / "model.joblib"

FEATURES: List[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with: python -m src.train"
        )
    return joblib.load(model_path)


def validate_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = {}
    for f in FEATURES:
        try:
            row[f] = float(payload[f])
        except Exception:
            raise ValueError(f"Feature '{f}' must be numeric. Got: {payload[f]!r}")

    return pd.DataFrame([row], columns=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser(description="California Housing CLI inference")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--json-file",
        type=Path,
        required=True,
        help="Path to JSON file containing features",
    )
    args = parser.parse_args()

    model = load_model(args.model_path)

    if not args.json_file.exists():
        raise FileNotFoundError(f"{args.json_file} does not exist.")

    payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    X = validate_payload(payload)

    pred = float(model.predict(X)[0])
    print(json.dumps({"prediction": pred, "model_path": str(args.model_path)}))



if __name__ == "__main__":
    main()
