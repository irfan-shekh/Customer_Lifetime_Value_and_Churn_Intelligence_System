"""
Automated tests for retention campaign ROI simulation.
"""

import pytest
from src.business.roi import simulate_retention_campaign, compare_campaign_scenarios


def test_simulate_retention_campaign_profitable():
    res = simulate_retention_campaign(
        eligible_count=1000,
        avg_monthly_charges=80.0,
        discount_pct=0.20,
        discount_duration_months=3,
        offer_acceptance_rate=0.50,
        baseline_churn_rate=0.60,
        incremental_uplift_rate=0.30,
        contribution_margin_pct=0.70,
        expected_remaining_lifetime_months=24,
    )
    assert res["eligible_customers"] == 1000
    assert res["offer_acceptances"] == 500
    assert res["incremental_retained_customers"] == int(500 * 0.60 * 0.30)  # 90
    assert res["net_benefit"] > 0
    assert res["roi_pct"] > 0
    assert res["is_profitable"] is True
    assert res["break_even_retained_customers"] < res["incremental_retained_customers"]


def test_simulate_retention_campaign_zero_uplift():
    # Zero uplift => net loss equal to campaign costs
    res = simulate_retention_campaign(
        eligible_count=500,
        avg_monthly_charges=50.0,
        discount_pct=0.20,
        incremental_uplift_rate=0.0,
    )
    assert res["incremental_retained_customers"] == 0
    assert res["total_retained_contribution"] == 0.0
    assert res["net_benefit"] < 0
    assert res["roi_pct"] == -100.0
    assert res["is_profitable"] is False


def test_compare_scenarios():
    scen = compare_campaign_scenarios(
        total_population=5000,
        high_risk_population=1200,
        avg_monthly_charges=65.0,
    )
    assert "no_campaign" in scen
    assert "broad_campaign" in scen
    assert "targeted_campaign" in scen
    # Targeted campaign should have substantially higher ROI than broad campaign
    assert scen["targeted_campaign"]["roi_pct"] > scen["broad_campaign"]["roi_pct"]
