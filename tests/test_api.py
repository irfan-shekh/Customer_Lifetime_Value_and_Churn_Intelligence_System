"""
Automated tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.services.api import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "ONLINE"
    assert "database" in data
    assert data["database"]["status"] == "HEALTHY"


def test_api_clv_estimate():
    response = client.post("/clv/estimate?monthly_charges=75.0&tenure=12&churn_probability=0.25")
    assert response.status_code == 200
    data = response.json()
    assert "discounted_clv" in data
    assert "expected_future_revenue" in data
    assert data["discounted_clv"] > 0


def test_api_predict_customer():
    payload = {
        "customer_id": "TEST-API-01",
        "gender": "Male",
        "senior_citizen": 0,
        "tenure": 5,
        "contract": "Month-to-month",
        "monthly_charges": 80.0,
        "total_charges": 400.0,
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    response = client.post("/predict/customer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert "clv" in data
    assert "recommendation" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
