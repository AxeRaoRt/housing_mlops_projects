# src/io_utils.py
import json
import os
from typing import Any, Dict

import joblib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(path: str, model: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
