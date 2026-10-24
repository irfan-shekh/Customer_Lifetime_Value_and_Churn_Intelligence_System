"""
Automated tests for churn classification modeling, threshold optimization, and probability ranges.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.churn import ChurnModelService, find_optimal_threshold, evaluate_model


def test_find_optimal_threshold():
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted probabilities well correlated with y
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    thresh, report = find_optimal_threshold(y_val, probs)
    assert 0.1 <= thresh <= 0.85
    assert "best_threshold" in report
    assert "net_value" in report["threshold_analysis"][0]


def test_evaluate_model_metrics():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    metrics = evaluate_model(y_true, probs, threshold=0.50)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_churn_service_predict_probabilities():
    service = ChurnModelService.load()
    test_sample = pd.DataFrame([{
        "customer_id": "TEST-01",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.0,
        "total_charges": 170.0,
    }])
    probs = service.predict_proba(test_sample)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0
    assert probs[0] > 0.40  # Month-to-month fiber optic short tenure has high churn risk
