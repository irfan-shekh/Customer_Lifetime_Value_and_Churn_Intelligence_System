"""
Business Decision Support and Policy Recommendation Engine.
Synthesizes churn probability, survival outlook, CLV valuation, and cluster segments
into concrete, prioritized operational actions for telecom business managers.
"""

from typing import Dict, Any, List


def get_customer_priority_tier(churn_probability: float, clv: float) -> str:
    """
    Ranks customer priority using a 2x2 Matrix:
    - High CLV + High Churn Risk -> Priority 1 (CRITICAL IMMEDIATE INTERVENTION)
    - High CLV + Low Churn Risk  -> Priority 2 (VIP RETENTION & NURTURING)
    - Low CLV  + High Churn Risk -> Priority 3 (LOW-COST AUTOMATED INTERVENTION)
    - Low CLV  + Low Churn Risk  -> Priority 4 (STANDARD BASELINE SERVICE)
    """
    is_high_churn = churn_probability >= 0.40
    is_high_clv = clv >= 600.0

    if is_high_churn and is_high_clv:
        return "Priority 1: Urgent VIP Intervention"
    elif not is_high_churn and is_high_clv:
        return "Priority 2: VIP Loyalty Nurturing"
    elif is_high_churn and not is_high_clv:
        return "Priority 3: Automated Digital Retention"
    else:
        return "Priority 4: Standard Maintenance"


def generate_prescriptive_recommendation(customer_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates tailored, actionable operational recommendations for nontechnical business managers.
    """
    prob = customer_data.get("churn_probability", 0.0) or 0.0
    clv = customer_data.get("discounted_clv", 0.0) or 0.0
    contract = str(customer_data.get("contract", "")).lower()
    internet = str(customer_data.get("internet_service", "")).lower()
    tech_sup = str(customer_data.get("tech_support", "")).lower()
    tenure = int(customer_data.get("tenure", 0))

    priority = get_customer_priority_tier(prob, clv)

    if priority.startswith("Priority 1"):
        rationale = (
            f"This customer carries high financial value (Estimated CLV: ${clv:,.2f}) "
            f"paired with an alarming {prob:.1%} churn probability. "
            "Allowing this customer to churn represents a direct material revenue loss."
        )
        if "month-to-month" in contract:
            action = (
                "Deploy proactive VIP outreach offering a 1-year contract lock-in with a 20% discount "
                "for the first 3 months plus complimentary premium Tech Support."
            )
        else:
            action = "Assign a dedicated customer success manager to conduct a satisfaction audit and resolve billing/service issues."
    elif priority.startswith("Priority 2"):
        rationale = (
            f"Highly valuable stable account (Estimated CLV: ${clv:,.2f}) with low churn risk ({prob:.1%}). "
            "Must be protected from competitor poaching."
        )
        action = "Enroll in customer appreciation program, provide loyalty credits, and offer early beta features."
    elif priority.startswith("Priority 3"):
        rationale = (
            f"Elevated churn likelihood ({prob:.1%}) but lower financial lifetime value (${clv:,.2f}). "
            "Expensive high-touch outreach is financially unprofitable."
        )
        action = "Send automated email/SMS discount voucher (e.g., $10 off next bill) or push digital self-service solutions."
    else:
        rationale = f"Stable, low-touch account with healthy retention characteristics (Tenure: {tenure}m)."
        action = "Maintain standard service levels and monitor for billing discrepancies."

    return {
        "priority_tier": priority,
        "business_rationale": rationale,
        "recommended_action": action,
    }
