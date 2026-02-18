import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import argparse
import json

from src.config import Settings
from src.io_utils import load_json, _mlflow_available, load_artifact_json_from_mlflow

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# PSI (Population Stability Index)
# ------------------------------------------------------------------ #

def _psi_bucket(expected_pct: float, actual_pct: float, eps: float = 1e-6) -> float:
    """PSI contribution from a single bucket."""
    e = max(expected_pct, eps)
    a = max(actual_pct, eps)
    return (a - e) * np.log(a / e)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two 1-D distributions.

    PSI < 0.10  → no significant shift
    PSI 0.10–0.25 → moderate shift (investigate)
    PSI > 0.25  → significant shift (retrain)
    """
    # Build bins from the expected distribution
    breakpoints = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected, breakpoints)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0].astype(float)
    actual_counts = np.histogram(actual, bins=bins)[0].astype(float)

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi = sum(_psi_bucket(e, a) for e, a in zip(expected_pct, actual_pct))
    return float(psi)


# ------------------------------------------------------------------ #
# Mean / Std shift
# ------------------------------------------------------------------ #

def compute_mean_shift(
    baseline_mean: float,
    baseline_std: float,
    actual_mean: float,
    threshold_z: float = 2.0,
) -> Tuple[float, bool]:
    """Return (z_score, is_drifted) comparing actual mean to baseline."""
    if baseline_std == 0:
        return (0.0, False)
    z = abs(actual_mean - baseline_mean) / baseline_std
    return (float(z), z > threshold_z)


# ------------------------------------------------------------------ #
# Full drift report
# ------------------------------------------------------------------ #

def load_baseline(
    local_path: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_model_name: Optional[str] = None,
    mlflow_stage: str = "Production",
) -> Dict[str, Any]:
    """Load the baseline from MLflow or local file.

    Priority: MLflow → local_path → Settings default.
    """
    # Try MLflow first
    if mlflow_tracking_uri and mlflow_model_name and _mlflow_available():
        try:
            baseline = load_artifact_json_from_mlflow(
                tracking_uri=mlflow_tracking_uri,
                model_name=mlflow_model_name,
                artifact_path="baseline.json",
                stage_or_version=mlflow_stage,
            )
            logger.info("Baseline loaded from MLflow (%s@%s)", mlflow_model_name, mlflow_stage)
            return baseline
        except Exception as e:
            logger.warning("Could not load baseline from MLflow: %s", e)

    # Local file fallback
    if local_path and os.path.exists(local_path):
        logger.info("Baseline loaded from local file: %s", local_path)
        return load_json(local_path)

    # Default path
    settings = Settings()
    default = os.path.join(settings.models_dir, "model_v1_baseline.json")
    if os.path.exists(default):
        logger.info("Baseline loaded from default path: %s", default)
        return load_json(default)

    raise FileNotFoundError("No baseline found (checked MLflow, local_path, and default).")


def detect_drift(
    live_df: pd.DataFrame,
    baseline: Dict[str, Any],
    features: Optional[List[str]] = None,
    psi_threshold: float = 0.25,
    z_threshold: float = 2.0,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Run drift detection on a batch of live data against the training baseline.

    Returns a report dict with per-feature and aggregate drift info.
    """
    baseline_features = baseline.get("features", {})
    if features is None:
        features = list(baseline_features.keys())

    report: Dict[str, Any] = {
        "n_samples": int(live_df.shape[0]),
        "n_features_checked": 0,
        "features_drifted_psi": [],
        "features_drifted_mean": [],
        "per_feature": {},
        "aggregate_psi": 0.0,
        "drift_detected": False,
    }

    psi_values = []

    for feat in features:
        if feat not in live_df.columns or feat not in baseline_features:
            continue

        report["n_features_checked"] += 1

        bl = baseline_features[feat]
        actual = live_df[feat].dropna().astype(float).values

        if len(actual) < 2:
            continue

        # --- PSI (requires synthetic "expected" from baseline quantiles) ---
        # Reconstruct approximate expected distribution from baseline stats
        expected_approx = np.random.normal(
            loc=bl["mean"], scale=max(bl["std"], 1e-8), size=max(len(actual), 1000)
        )
        psi = compute_psi(expected_approx, actual, n_bins=n_bins)
        psi_values.append(psi)

        # --- Mean shift ---
        z_score, mean_drifted = compute_mean_shift(
            baseline_mean=bl["mean"],
            baseline_std=bl["std"],
            actual_mean=float(actual.mean()),
            threshold_z=z_threshold,
        )

        feat_report = {
            "psi": round(psi, 6),
            "psi_drifted": psi > psi_threshold,
            "actual_mean": round(float(actual.mean()), 6),
            "baseline_mean": round(bl["mean"], 6),
            "z_score": round(z_score, 4),
            "mean_drifted": mean_drifted,
        }
        report["per_feature"][feat] = feat_report

        if psi > psi_threshold:
            report["features_drifted_psi"].append(feat)
        if mean_drifted:
            report["features_drifted_mean"].append(feat)

    if psi_values:
        report["aggregate_psi"] = round(float(np.mean(psi_values)), 6)

    report["drift_detected"] = (
        len(report["features_drifted_psi"]) > 0
        or len(report["features_drifted_mean"]) > 0
    )

    return report


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

def main():

    settings = Settings()

    parser = argparse.ArgumentParser(description="Drift detection against training baseline")
    parser.add_argument("--data-path", required=True, help="Path to live/new data CSV")
    parser.add_argument("--baseline-path", default=None, help="Local baseline JSON (fallback)")
    parser.add_argument("--psi-threshold", type=float, default=0.25)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--output", default=None, help="Optional output JSON path for report")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    live_df = pd.read_csv(args.data_path)

    baseline = load_baseline(
        local_path=args.baseline_path,
        mlflow_tracking_uri=None if args.no_mlflow else settings.mlflow_tracking_uri,
        mlflow_model_name=None if args.no_mlflow else settings.mlflow_registered_model_name,
    )

    report = detect_drift(
        live_df=live_df,
        baseline=baseline,
        psi_threshold=args.psi_threshold,
        z_threshold=args.z_threshold,
    )

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        from src.io_utils import save_json
        save_json(args.output, report)
        print(f"\nReport saved to: {args.output}")

    if report["drift_detected"]:
        print("\n⚠️  DRIFT DETECTED")
        if report["features_drifted_psi"]:
            print(f"  PSI drift:  {report['features_drifted_psi']}")
        if report["features_drifted_mean"]:
            print(f"  Mean drift: {report['features_drifted_mean']}")
    else:
        print("\n✅ No significant drift detected")


if __name__ == "__main__":
    main()



# """
# Data drift detection module.

# Compares live/incoming feature distributions against the training baseline
# stored in MLflow artifacts or local JSON files.

# Supports:
#   - Population Stability Index (PSI) per feature
#   - Simple mean/std shift detection
#   - Aggregated drift score for alerting
# """
