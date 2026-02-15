# src/validate_data.py
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from src.config import Settings


REQUIRED_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]


def validate(df: pd.DataFrame) -> dict:
    """Return a validation report dict. Raise ValueError if blocking checks fail."""
    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # --- Schema checks ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    report["checks"]["missing_columns"] = missing
    report["checks"]["extra_columns"] = extra

    if missing:
        report["errors"].append(f"Missing required columns: {missing}")

    # --- Null checks ---
    null_counts = df.isna().sum().to_dict()
    null_total = int(sum(null_counts.values()))
    report["checks"]["null_total"] = null_total
    report["checks"]["null_counts_top"] = dict(
        sorted(null_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    )
    if null_total > 0:
        report["errors"].append(f"Found {null_total} missing values (NaNs).")

    # --- Type checks (soft, warnings) ---
    # Expect numeric for all columns in this dataset
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    report["checks"]["non_numeric_columns"] = non_numeric
    if non_numeric:
        report["warnings"].append(f"Non-numeric columns detected: {non_numeric}")

    # --- Range checks ---
    if "Amount" in df.columns:
        neg_amount = int((df["Amount"] < 0).sum())
        report["checks"]["negative_amount_count"] = neg_amount
        if neg_amount > 0:
            report["errors"].append(f"Amount has {neg_amount} negative values (< 0).")

    if "Time" in df.columns:
        neg_time = int((df["Time"] < 0).sum())
        report["checks"]["negative_time_count"] = neg_time
        if neg_time > 0:
            report["errors"].append(f"Time has {neg_time} negative values (< 0).")

    # --- Target checks ---
    if "Class" in df.columns:
        unique = sorted(df["Class"].dropna().unique().tolist())
        report["checks"]["target_unique_values"] = unique
        # Must be binary {0,1}
        if set(unique) - {0, 1}:
            report["errors"].append(f"Target Class must be binary 0/1. Got: {unique}")

        fraud_rate = float(df["Class"].mean())
        report["checks"]["fraud_rate"] = fraud_rate
        # Not an error, but useful warning if something is odd
        if fraud_rate <= 0 or fraud_rate >= 0.5:
            report["warnings"].append(
                f"Fraud rate looks unusual: {fraud_rate:.6f} (expected very low for this dataset)."
            )

    # Blocking decision
    report["checks"]["passed"] = (len(report["errors"]) == 0)
    if report["errors"]:
        raise ValueError("Validation failed: " + " | ".join(report["errors"]))
    return report


def main():
    settings = Settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=settings.data_path)
    parser.add_argument("--report-path", default=None, help="Optional explicit report output path")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    os.makedirs(settings.reports_dir, exist_ok=True)
    report_path = args.report_path or os.path.join(settings.reports_dir, "data_validation_report.json")

    try:
        report = validate(df)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Data validation passed. Report written to: {report_path}")
        sys.exit(0)
    except Exception as e:
        # Even on failure, write a report with errors captured
        fail_report = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "passed": False,
            "error": str(e),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(fail_report, f, indent=2)
        print(f"❌ Data validation FAILED. Report written to: {report_path}")
        print(f"Reason: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
