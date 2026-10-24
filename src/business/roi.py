"""
Retention Campaign Financial ROI Simulation Module.
Transparent financial formulas for evaluating customer retention campaign profitability,
comparing untargeted mass discounts vs high-precision ML-targeted interventions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def simulate_retention_campaign(
    eligible_count: int,
    avg_monthly_charges: float,
    discount_pct: float = 0.20,
    discount_duration_months: int = 3,
    offer_acceptance_rate: float = 0.45,
    baseline_churn_rate: float = 0.55,
    incremental_uplift_rate: float = 0.25,
    contribution_margin_pct: float = 0.65,
    expected_remaining_lifetime_months: int = 18,
    fixed_operational_cost: float = 1000.0,
    budget_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulates the financial impact of a retention campaign.
    """
    # 1. Budget constraint check
    cost_per_acceptor = avg_monthly_charges * discount_pct * discount_duration_months
    effective_eligible = eligible_count

    if budget_limit is not None and budget_limit > 0:
        max_acceptors = max(1, int((budget_limit - fixed_operational_cost) / max(cost_per_acceptor, 1.0)))
        max_eligible = int(max_acceptors / max(offer_acceptance_rate, 0.01))
        effective_eligible = min(effective_eligible, max_eligible)

    # 2. Volumes
    expected_acceptances = int(effective_eligible * offer_acceptance_rate)
    
    # Incremental saved customers (would have churned without offer, but stay because of offer)
    # Incremental retained = Expected Acceptances * Baseline Churn Rate * Incremental Uplift
    incremental_retained_customers = int(expected_acceptances * baseline_churn_rate * incremental_uplift_rate)

    # 3. Campaign Costs
    # Discount cost given to all customers who accepted the offer
    total_discount_cost = expected_acceptances * cost_per_acceptor
    total_campaign_cost = total_discount_cost + fixed_operational_cost

    # 4. Retained Gross Margin / Contribution
    # Future gross profit preserved from incremental retained customers over remaining lifetime
    margin_per_customer_lifetime = (
        avg_monthly_charges * contribution_margin_pct * expected_remaining_lifetime_months
    )
    total_retained_contribution = incremental_retained_customers * margin_per_customer_lifetime

    # 5. Net Financial Benefit and ROI
    net_benefit = total_retained_contribution - total_campaign_cost
    roi_pct = (net_benefit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

    # 6. Break-Even Analysis
    # Break-even incremental customers needed to cover campaign costs
    break_even_customers = (
        int(np.ceil(total_campaign_cost / margin_per_customer_lifetime))
        if margin_per_customer_lifetime > 0
        else 0
    )
    
    break_even_uplift = (
        (break_even_customers / max(1, expected_acceptances * baseline_churn_rate))
        if (expected_acceptances * baseline_churn_rate) > 0
        else 0.0
    )

    return {
        "eligible_customers": int(effective_eligible),
        "offer_acceptances": expected_acceptances,
        "incremental_retained_customers": incremental_retained_customers,
        "cost_per_acceptor": round(cost_per_acceptor, 2),
        "total_campaign_cost": round(total_campaign_cost, 2),
        "total_discount_cost": round(total_discount_cost, 2),
        "fixed_operational_cost": round(fixed_operational_cost, 2),
        "margin_per_retained_customer": round(margin_per_customer_lifetime, 2),
        "total_retained_contribution": round(total_retained_contribution, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_pct": round(roi_pct, 1),
        "break_even_retained_customers": break_even_customers,
        "break_even_uplift_pct": round(break_even_uplift * 100, 1),
        "is_profitable": bool(net_benefit > 0),
    }


def compare_campaign_scenarios(
    total_population: int,
    high_risk_population: int,
    avg_monthly_charges: float,
    margin_pct: float = 0.65,
) -> Dict[str, Dict[str, Any]]:
    """
    Compares 3 standard scenarios:
    1. No Campaign: Baseline scenario (0 cost, 0 incremental retain).
    2. Broad Mass Discount: 20% discount offered indiscriminately to all customers.
    3. Targeted ML Campaign: 20% discount offered selectively to high-risk / high-CLV customers.
    """
    # Scenario A: No Campaign
    no_camp = {
        "scenario_name": "No Retention Action (Baseline)",
        "targeted_customers": 0,
        "campaign_cost": 0.0,
        "retained_customers": 0,
        "retained_contribution": 0.0,
        "net_benefit": 0.0,
        "roi_pct": 0.0,
    }

    # Scenario B: Broad Mass Campaign
    broad = simulate_retention_campaign(
        eligible_count=total_population,
        avg_monthly_charges=avg_monthly_charges,
        discount_pct=0.20,
        discount_duration_months=3,
        offer_acceptance_rate=0.50,
        baseline_churn_rate=0.26,  # Overall population churn
        incremental_uplift_rate=0.15,
        contribution_margin_pct=margin_pct,
        expected_remaining_lifetime_months=18,
    )
    broad_res = {
        "scenario_name": "Broad Mass Campaign (Untargeted)",
        "targeted_customers": broad["eligible_customers"],
        "campaign_cost": broad["total_campaign_cost"],
        "retained_customers": broad["incremental_retained_customers"],
        "retained_contribution": broad["total_retained_contribution"],
        "net_benefit": broad["net_benefit"],
        "roi_pct": broad["roi_pct"],
    }

    # Scenario C: Targeted ML Campaign
    targeted = simulate_retention_campaign(
        eligible_count=high_risk_population,
        avg_monthly_charges=avg_monthly_charges * 1.15,  # High risk tends to have higher monthly spend
        discount_pct=0.20,
        discount_duration_months=3,
        offer_acceptance_rate=0.45,
        baseline_churn_rate=0.65,  # High-risk segment churn
        incremental_uplift_rate=0.28,  # Persuadable focus
        contribution_margin_pct=margin_pct,
        expected_remaining_lifetime_months=18,
    )
    targeted_res = {
        "scenario_name": "Targeted ML Campaign (High Risk & High CLV)",
        "targeted_customers": targeted["eligible_customers"],
        "campaign_cost": targeted["total_campaign_cost"],
        "retained_customers": targeted["incremental_retained_customers"],
        "retained_contribution": targeted["total_retained_contribution"],
        "net_benefit": targeted["net_benefit"],
        "roi_pct": targeted["roi_pct"],
    }

    return {
        "no_campaign": no_camp,
        "broad_campaign": broad_res,
        "targeted_campaign": targeted_res,
    }
