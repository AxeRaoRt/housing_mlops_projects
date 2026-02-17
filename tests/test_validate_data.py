# tests/test_validate_data.py
"""Unit tests for src.validate_data — uses synthetic DataFrames (no real CSV needed)."""
import pytest
import numpy as np
import pandas as pd

from src.validate_data import validate, REQUIRED_COLUMNS


def _make_valid_df(n: int = 100) -> pd.DataFrame:
    """Generate a minimal valid DataFrame matching the expected schema."""
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in REQUIRED_COLUMNS if col not in ("Time", "Amount", "Class")}
    data["Time"] = rng.uniform(0, 172_000, n)
    data["Amount"] = rng.uniform(0, 500, n)
    data["Class"] = rng.choice([0, 1], n, p=[0.95, 0.05])
    return pd.DataFrame(data)


# ---------- Happy path ----------

def test_validate_passes_on_valid_data():
    df = _make_valid_df()
    report = validate(df)
    assert report["checks"]["passed"] is True
    assert report["rows"] == 100
    assert len(report["errors"]) == 0


def test_validate_reports_correct_shape():
    df = _make_valid_df(200)
    report = validate(df)
    assert report["rows"] == 200
    assert report["cols"] == len(REQUIRED_COLUMNS)


# ---------- Schema checks ----------

def test_validate_fails_on_missing_columns():
    df = _make_valid_df().drop(columns=["V1", "V2"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df)


def test_validate_detects_extra_columns():
    df = _make_valid_df()
    df["ExtraCol"] = 0
    report = validate(df)
    assert "ExtraCol" in report["checks"]["extra_columns"]
    # Extra columns are not blocking
    assert report["checks"]["passed"] is True


# ---------- Null checks ----------

def test_validate_fails_on_nulls():
    df = _make_valid_df()
    df.loc[0, "V1"] = np.nan
    df.loc[1, "V14"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate(df)


# ---------- Range checks ----------

def test_validate_fails_on_negative_amount():
    df = _make_valid_df()
    df.loc[0, "Amount"] = -10.0
    with pytest.raises(ValueError, match="negative values"):
        validate(df)


def test_validate_fails_on_negative_time():
    df = _make_valid_df()
    df.loc[0, "Time"] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        validate(df)


# ---------- Target checks ----------

def test_validate_fails_on_non_binary_target():
    df = _make_valid_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="binary 0/1"):
        validate(df)


def test_validate_warns_on_high_fraud_rate():
    df = _make_valid_df()
    df["Class"] = 1  # 100% fraud
    # Should warn but not block — the validate function raises because fraud_rate >= 0.5
    # Actually: fraud_rate warning is non-blocking, but let's check
    # Looking at the code: if fraud_rate >= 0.5 it adds a warning, not an error
    report = validate(df)
    assert any("unusual" in w.lower() for w in report["warnings"])
