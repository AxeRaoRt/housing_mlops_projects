import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
import mlflow
# import pandas as pd
from mlflow.tracking import MlflowClient

import joblib

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Local I/O helpers 
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# MLflow helpers
# ------------------------------------------------------------------ #

def _mlflow_available() -> bool:
    """Return True if mlflow is installed."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False

def log_training_run_to_mlflow(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    schema: Dict[str, Any],
    baseline: Dict[str, Any],
    model_version_tag: str = "v1",
    input_example: Optional[Any] = None,
) -> str:
    """Log a full training run to MLflow and register the model.

    Returns the MLflow run_id.
    """

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # --- Parameters ---
        mlflow.log_params(params)

        # --- Scalar metrics (flatten for MLflow) ---
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)

        # --- Tags ---
        mlflow.set_tag("model_version_tag", model_version_tag)

        # --- Full metrics JSON as artifact ---
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            mlflow.log_artifact(metrics_path)

        # --- Schema as artifact ---
        mlflow.log_dict(schema, artifact_file="schema.json")

        # --- Baseline as artifact ---
        mlflow.log_dict(baseline, artifact_file="baseline.json")

        # --- Préparer un input_example propre ---
        safe_example = None
        if input_example is not None:
            if hasattr(input_example, "to_dict"):
                safe_example = input_example
            else:
                safe_example = input_example

        # --- Model (register in Model Registry) ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=safe_example,
        )

        # --- AUTO-PROMOTE to Production ---
        try:
            client = MlflowClient()
            latest_versions = client.get_latest_versions(registered_model_name, stages=["None"])
            if latest_versions:
                new_version = latest_versions[0].version
                # On utilise un ALIAS au lieu d'un STAGE
                client.set_registered_model_alias(
                    name=registered_model_name, 
                    alias="Production", 
                    version=new_version
                )
                logger.info(f"Alias 'Production' set to version {new_version}")
        except Exception as e:
            logger.warning("Alias promotion failed: %s", e)

        logger.info("MLflow run logged: run_id=%s, experiment=%s", run.info.run_id, experiment_name)
        return run.info.run_id


def load_model_from_mlflow(tracking_uri: str, model_name: str, stage_or_version: str = "Production"):
    mlflow.set_tracking_uri(tracking_uri)
    
    if stage_or_version.isdigit():
        model_uri = f"models:/{model_name}/{stage_or_version}"
    else:
        # Syntaxe pour les Aliases (@)
        model_uri = f"models:/{model_name}@{stage_or_version}"

    logger.info("Loading model from MLflow: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def load_artifact_json_from_mlflow(
    tracking_uri: str,
    model_name: str,
    artifact_path: str,
    stage_or_version: str = "Production",
) -> Dict[str, Any]:
    """Download a JSON artifact associated with a registered model version."""

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if stage_or_version.isdigit():
        mv = client.get_model_version(model_name, int(stage_or_version))
    else:
        # Chercher par ALIAS
        try:
            mv = client.get_model_version_by_alias(model_name, stage_or_version)
        except:
            versions = client.get_latest_versions(model_name, stages=[stage_or_version])
            if not versions: 
                raise ValueError(f"Not found: {model_name}@{stage_or_version}")
            mv = versions[0]

    run_id = mv.run_id
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )

    return load_json(local_path)
