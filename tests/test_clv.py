"""
Automated tests for Customer Lifetime Value calculations.
"""

import pytest
import pandas as pd
import numpy as np

from src.models.clv import CLVEstimator
from src.models.survival import SurvivalAnalysisService


def test_clv_single_calculation():
    surv = SurvivalAnalysisService.load()
    est = CLVEstimator(surv)
    
    res = est.calculate_customer_clv(
        monthly_charges=70.0,
        current_tenure=12,
        churn_probability=0.30,
        horizon_months=24,
        margin=0.65,
        discount_rate=0.10,
    )
    assert res["discounted_clv"] > 0
    assert res["expected_future_revenue"] > res["discounted_clv"]
    assert res["revenue_at_risk"] == round(res["discounted_clv"] * 0.30, 2)


def test_clv_zero_charges_edge_case():
    surv = SurvivalAnalysisService.load()
    est = CLVEstimator(surv)
    
    res = est.calculate_customer_clv(
        monthly_charges=0.0,
        current_tenure=0,
        churn_probability=0.50,
    )
    assert res["discounted_clv"] == 0.0
    assert res["revenue_at_risk"] == 0.0


def test_clv_margin_sensitivity():
    surv = SurvivalAnalysisService.load()
    est = CLVEstimator(surv)
    
    low_margin = est.calculate_customer_clv(monthly_charges=100.0, current_tenure=10, churn_probability=0.2, margin=0.40)
    high_margin = est.calculate_customer_clv(monthly_charges=100.0, current_tenure=10, churn_probability=0.2, margin=0.80)
    
    assert high_margin["discounted_clv"] > low_margin["discounted_clv"]
    assert high_margin["expected_future_revenue"] == low_margin["expected_future_revenue"]
