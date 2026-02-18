import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Settings
from src.io_utils import save_json, save_model, log_training_run_to_mlflow, _mlflow_available


REQUIRED_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def compute_baseline(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    # Simple baseline: mean/std + quantiles per feature (enough for drift checks later)
    baseline = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(df.shape[0]),
        "features": {},
    }
    for c in feature_cols:
        s = df[c].astype(float)
        baseline["features"][c] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "q05": float(s.quantile(0.05)),
            "q50": float(s.quantile(0.50)),
            "q95": float(s.quantile(0.95)),
        }
    return baseline


def main():
    settings = Settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=settings.data_path)
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (local-only mode)",
    )
    args = parser.parse_args()

    use_mlflow = (not args.no_mlflow) and _mlflow_available()

    set_seeds(settings.seed)

    df = pd.read_csv(args.data_path)

    # Basic schema check (training should fail fast)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_col = settings.target_col
    feature_cols = [c for c in REQUIRED_COLUMNS if c != target_col]

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=settings.seed,
        stratify=y,
    )

    # Simple, production-friendly baseline model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)),
        ]
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    # Predict probabilities for metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "model_version": args.model_version,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": settings.seed,
        "rows_train": int(X_train.shape[0]),
        "rows_test": int(X_test.shape[0]),
        "train_seconds": float(train_seconds),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    schema = {
        "target": target_col,
        "features": feature_cols,
        "n_features": len(feature_cols),
    }

    baseline = compute_baseline(X_train, feature_cols)

    # ------------------------------------------------------------------ #
    # Save artifacts locally (always — serves as fallback)
    # ------------------------------------------------------------------ #
    os.makedirs(settings.models_dir, exist_ok=True)
    model_path = os.path.join(settings.models_dir, f"model_{args.model_version}.joblib")
    metrics_path = os.path.join(settings.models_dir, f"model_{args.model_version}_metrics.json")
    schema_path = os.path.join(settings.models_dir, f"model_{args.model_version}_schema.json")
    baseline_path = os.path.join(settings.models_dir, f"model_{args.model_version}_baseline.json")

    save_model(model_path, model)
    save_json(metrics_path, metrics)
    save_json(schema_path, schema)
    save_json(baseline_path, baseline)

    print("✅ Training complete — local artifacts saved")
    print(f"  Model:    {model_path}")
    print(f"  Metrics:  {metrics_path}")
    print(f"  Schema:   {schema_path}")
    print(f"  Baseline: {baseline_path}")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # Log to MLflow (experiment tracking + model registry)
    # ------------------------------------------------------------------ #
    if use_mlflow:
        try:
            # input_example = X_test.iloc[:1].to_dict(orient="list")

            run_id = log_training_run_to_mlflow(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
                registered_model_name=settings.mlflow_registered_model_name,
                model=model,
                params={
                    "test_size": args.test_size,
                    "seed": settings.seed,
                    "max_iter": 200,
                    "class_weight": "balanced",
                    "n_features": len(feature_cols),
                    "data_path": args.data_path,
                },
                metrics=metrics,
                schema=schema,
                baseline=baseline,
                model_version_tag=args.model_version,
                input_example=X_test.head(1),
            )
            print(f"\n🔬 MLflow run logged: {run_id}")
            print(f"   Tracking UI: {settings.mlflow_tracking_uri}")
            print(f"   Experiment:  {settings.mlflow_experiment_name}")
            print(f"   Registered:  {settings.mlflow_registered_model_name}")
        except Exception as e:
            print(f"\n⚠️  MLflow logging failed (local artifacts are still saved): {e}")
    else:
        if args.no_mlflow:
            print("\nℹ️  MLflow logging disabled (--no-mlflow flag)")
        else:
            print("\nℹ️  MLflow not available install with: pip install mlflow")


if __name__ == "__main__":
    main()
