# tests/test_train.py
"""Smoke tests for src.train — trains on a tiny synthetic dataset."""
import os
import json
# import tempfile
# import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.validate_data import REQUIRED_COLUMNS


def _make_train_csv(path: str, n: int = 200) -> None:
    """Write a small but valid creditcard-like CSV for training."""
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in REQUIRED_COLUMNS if col not in ("Time", "Amount", "Class")}
    data["Time"] = rng.uniform(0, 172_000, n)
    data["Amount"] = rng.uniform(0, 500, n)
    # Ensure at least some fraud (stratified split needs both classes)
    labels = np.zeros(n, dtype=int)
    labels[: max(10, n // 20)] = 1
    rng.shuffle(labels)
    data["Class"] = labels
    pd.DataFrame(data).to_csv(path, index=False)


class TestTrainSmoke:
    """Train on a tiny dataset and verify artifacts are created."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path):
        self.tmp = tmp_path
        self.data_path = str(tmp_path / "mini.csv")
        self.models_dir = str(tmp_path / "models")
        os.makedirs(self.models_dir, exist_ok=True)
        _make_train_csv(self.data_path, n=200)

    def test_train_creates_artifacts(self):
        """Run src.train.main() via subprocess to avoid global state issues."""
        
        result = subprocess.run(
            [
                "python", "-m", "src.train",
                "--data-path", self.data_path,
                "--model-version", "test",
                "--no-mlflow",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "MODELS_DIR": self.models_dir},
            cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        # Check artifacts exist
        for suffix in [".joblib", "_metrics.json", "_schema.json", "_baseline.json"]:
            artifact = os.path.join(self.models_dir, f"model_test{suffix}")
            assert os.path.exists(artifact), f"Missing artifact: {artifact}"

        # Validate metrics content
        metrics_path = os.path.join(self.models_dir, "model_test_metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert "pr_auc" in metrics
        assert metrics["model_version"] == "test"

    def test_train_schema_has_features(self):
        """Verify the schema lists all expected features."""

        result = subprocess.run(
            [
                "python", "-m", "src.train",
                "--data-path", self.data_path,
                "--model-version", "test2",
                "--no-mlflow",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "MODELS_DIR": self.models_dir},
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0

        schema_path = os.path.join(self.models_dir, "model_test2_schema.json")
        with open(schema_path) as f:
            schema = json.load(f)
        assert schema["target"] == "Class"
        assert len(schema["features"]) == 30  # Time + Amount + V1..V28
