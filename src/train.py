import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.common.utils import load_california_housing

SEED = 42
MODEL_VERSION = "v1"

MODEL_DIR = Path("models") / MODEL_VERSION
ARTIFACT_DIR = Path("artifacts") / MODEL_VERSION

MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    print("Starting training pipeline...")

    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_california_housing()
    X, y = ds.X, ds.y

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {ds.feature_names}")



    
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "seed": SEED,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")


if __name__ == "__main__":
    main()
