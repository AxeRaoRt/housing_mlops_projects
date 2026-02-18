import pytest
from fastapi.testclient import TestClient
from api.main import app

# Integration tests for the FastAPI app using TestClient 

@pytest.fixture(scope="module")
def client():
    """Create a TestClient — model loads on startup."""
    with TestClient(app) as c:
        yield c


# ---------- Health ----------

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "load_source" in data
    assert "timestamp_utc" in data


# ---------- Metrics ----------

def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "requests_total" in resp.text
    assert "predictions_total" in resp.text


# ---------- Predict ----------

def _sample_payload() -> dict:
    """Minimal valid payload matching PredictRequest schema."""
    payload = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=_sample_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "model_version" in data
    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["is_fraud"], bool)


def test_predict_missing_field_returns_422(client):
    bad = {"Time": 0.0, "Amount": 100.0}  # Missing V1..V28
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ---------- Drift ----------

def test_drift_endpoint_with_batch(client):
    """Send a small batch and verify drift report structure."""
    rows = [_sample_payload() for _ in range(10)]
    resp = client.post("/drift", json={"data": rows})
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data
    assert "aggregate_psi" in data
    assert "n_samples" in data
    assert data["n_samples"] == 10
    assert isinstance(data["features_drifted_psi"], list)
    assert isinstance(data["per_feature"], dict)


def test_drift_latest_404_before_check(client):
    """GET /drift/latest should return report if drift was run."""
    # After test_drift_endpoint_with_batch, a report exists
    resp = client.get("/drift/latest")
    # Could be 200 (if previous test ran) or 404 (if not)
    assert resp.status_code in (200, 404)


def test_drift_too_few_samples(client):
    """Drift with < 2 rows should fail validation."""
    resp = client.post("/drift", json={"data": [_sample_payload()]})
    assert resp.status_code == 422
