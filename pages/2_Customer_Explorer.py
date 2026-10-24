"""
Customer Explorer Page.
Enables 360-degree customer search, risk tier filtering, local SHAP factor breakdown,
conditional survival outlook, and prescriptive business retention recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import get_engine
from src.database.repository import DatabaseRepository
from src.models.shap_explainer import ShapExplanationService, CAUSAL_DISCLAIMER
from src.business.recommendations import generate_prescriptive_recommendation

st.set_page_config(page_title="Customer Explorer", page_icon="🔍", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Customer Explorer</h1>
    <p>360-Degree Individual Account Audit, SHAP Risk Factors & Prescriptive Recommendations</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

repo = DatabaseRepository()
engine = get_engine()

if engine is None or not repo.is_connected:
    st.error("⚠️ PostgreSQL Database is currently disconnected.")
    st.info("Please ensure PostgreSQL 18 is running and verify DATABASE_URL in .env.")
    st.stop()

# Sidebar Search & Filters
st.sidebar.header("Customer Filter & Search")
risk_filter = st.sidebar.selectbox("Filter by Risk Tier", ["All Tiers", "Critical", "High", "Medium", "Low"])
contract_filter = st.sidebar.selectbox("Filter by Contract", ["All Contracts", "Month-to-month", "One year", "Two year"])

# Query candidate customers
where_clauses = []
if risk_filter != "All Tiers":
    where_clauses.append(f"p.risk_tier = '{risk_filter}'")
if contract_filter != "All Contracts":
    where_clauses.append(f"c.contract = '{contract_filter}'")

where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

query = f"""
    SELECT c.customer_id, p.risk_tier, p.churn_probability, v.discounted_clv, c.contract, s.monthly_charges, s.tenure
    FROM customers c
    JOIN customer_snapshots s ON c.customer_id = s.customer_id
    LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
    LEFT JOIN customer_lifetime_values v ON c.customer_id = v.customer_id
    {where_str}
    ORDER BY p.churn_probability DESC
    LIMIT 200
"""
df_list = pd.read_sql(query, engine)

if df_list.empty:
    st.warning("No customers match the selected filter criteria.")
    st.stop()

# Search selection
selected_id = st.selectbox(
    "Select Customer ID to Inspect",
    options=df_list["customer_id"].tolist(),
    format_func=lambda x: f"{x} (Churn Risk: {df_list.loc[df_list['customer_id']==x, 'churn_probability'].values[0]:.1%}, Tier: {df_list.loc[df_list['customer_id']==x, 'risk_tier'].values[0]})",
)

# Fetch customer 360 profile
profile = repo.get_customer_profile(selected_id)

if not profile:
    st.error(f"Customer {selected_id} not found in database.")
    st.stop()

# Header Metrics Row
prob = profile.get("churn_probability", 0.0) or 0.0
clv = profile.get("discounted_clv", 0.0) or 0.0
rev_at_risk = profile.get("revenue_at_risk", 0.0) or 0.0
tier = profile.get("risk_tier", "Unscored")

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Churn Risk Probability", f"{prob:.1%}", f"Risk Tier: {tier}", is_danger=(prob >= 0.50))
with c2:
    render_metric_card("Estimated CLV", f"${clv:,.2f}", "24M Discounted Forward Margin")
with c3:
    render_metric_card("Revenue At Risk", f"${rev_at_risk:,.2f}", "Expected Financial Exposure", is_danger=True)
with c4:
    render_metric_card("Customer Segment", profile.get("cluster_label", "Unassigned"), f"Tenure: {profile['tenure']} Months")

st.markdown("---")

# Prescriptive Action Banner
recom = generate_prescriptive_recommendation(profile)
st.subheader("🎯 Prescriptive Action Plan")
col_recom1, col_recom2 = st.columns([1, 2])
with col_recom1:
    st.markdown(f"**Priority Level:** `{recom['priority_tier']}`")
    st.markdown(f"**Business Rationale:** {recom['business_rationale']}")
with col_recom2:
    st.success(f"**Recommended Retention Action:**\n\n{recom['recommended_action']}")

st.markdown("---")

# Attributes & SHAP Breakdown
col_attr, col_shap = st.columns([1, 1])

with col_attr:
    st.subheader("📋 Account & Service Portfolio")
    attr_df = pd.DataFrame([
        {"Attribute": "Contract Type", "Value": profile.get("contract")},
        {"Attribute": "Monthly Charges", "Value": f"${profile.get('monthly_charges'):.2f}"},
        {"Attribute": "Historical Total Charges", "Value": f"${profile.get('total_charges'):.2f}"},
        {"Attribute": "Tenure with Provider", "Value": f"{profile.get('tenure')} months"},
        {"Attribute": "Internet Service", "Value": profile.get("internet_service")},
        {"Attribute": "Payment Method", "Value": profile.get("payment_method")},
        {"Attribute": "Partner / Dependents", "Value": f"{profile.get('partner')} / {profile.get('dependents')}"},
        {"Attribute": "Senior Citizen", "Value": "Yes" if profile.get("senior_citizen") == 1 else "No"},
    ])
    st.dataframe(attr_df, use_container_width=True, hide_index=True)

    # Survival Probabilities
    st.subheader("⏱️ Retention Horizon Projections")
    surv_df = pd.DataFrame([
        {"Horizon": "Next 12 Months", "Conditional Retention Probability": f"{profile.get('prob_active_12m', 0.0):.1%}"},
        {"Horizon": "Next 24 Months", "Conditional Retention Probability": f"{profile.get('prob_active_24m', 0.0):.1%}"},
    ])
    st.dataframe(surv_df, use_container_width=True, hide_index=True)

with col_shap:
    st.subheader("🔍 Individual SHAP Risk Drivers")
    st.caption("Marginal contribution to this customer's predicted risk score:")

    # Calculate real-time SHAP explanation
    try:
        shap_service = ShapExplanationService()
        raw_cust_df = pd.DataFrame([{
            "customer_id": profile["customer_id"],
            "gender": profile["gender"],
            "senior_citizen": profile["senior_citizen"],
            "partner": profile["partner"],
            "dependents": profile["dependents"],
            "tenure": profile["tenure"],
            "phone_service": profile["phone_service"],
            "contract": profile["contract"],
            "payment_method": profile["payment_method"],
            "monthly_charges": profile["monthly_charges"],
            "total_charges": profile["total_charges"],
            "internet_service": profile["internet_service"],
        }])
        explanation = shap_service.explain_single_customer(raw_cust_df, top_k=4)
        
        # Plot SHAP Drivers
        pos_drivers = pd.DataFrame(explanation["risk_increasing_factors"])
        neg_drivers = pd.DataFrame(explanation["risk_decreasing_factors"])
        
        combined_drivers = pd.concat([pos_drivers, neg_drivers])
        if not combined_drivers.empty:
            fig_shap = px.bar(
                combined_drivers,
                x="shap_impact",
                y="feature",
                orientation="h",
                color="shap_impact",
                color_continuous_scale=["#10B981", "#E2E8F0", "#EF4444"],
                title="Top Predictive Factors (SHAP Feature Attribution)",
                labels={"shap_impact": "SHAP Risk Contribution (+ Increase / - Decrease)", "feature": "Feature"},
            )
            fig_shap.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.info("Loading precomputed risk attribution...")

    st.warning(f"⚠️ {CAUSAL_DISCLAIMER}")
