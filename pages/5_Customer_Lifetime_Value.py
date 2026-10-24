"""
Customer Lifetime Value (CLV) Page.
Actuarial survival-discounted cash flow modeling, financial scenario adjustments,
and revenue at risk segment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import get_engine
from src.models.clv import CLVEstimator, CLV_METHODOLOGY_DISCLAIMER
from src.models.survival import SurvivalAnalysisService

st.set_page_config(page_title="Customer Lifetime Value", page_icon="💰", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Customer Lifetime Value (CLV)</h1>
    <p>Actuarial Discounted Cash Flow Valuation, Forward Contribution Margins & Revenue at Risk</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

# Financial parameter controls in sidebar
st.sidebar.header("CLV Financial Parameters")
horizon_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=6, max_value=60, value=24, step=6)
contribution_margin = st.sidebar.slider("Gross Contribution Margin (%)", min_value=30, max_value=95, value=65, step=5) / 100.0
discount_rate = st.sidebar.slider("Annual WACC / Discount Rate (%)", min_value=0, max_value=25, value=10, step=1) / 100.0

engine = get_engine()
if engine is None:
    st.error("⚠️ PostgreSQL Database is currently disconnected.")
    st.info("Please ensure PostgreSQL 18 is running and verify DATABASE_URL in .env.")
    st.stop()

df_clv_raw = pd.read_sql("""
    SELECT c.customer_id, c.contract, c.internet_service, s.tenure, s.monthly_charges, s.total_charges, p.churn_probability, p.risk_tier
    FROM customers c
    JOIN customer_snapshots s ON c.customer_id = s.customer_id
    LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
""", engine)

if df_clv_raw.empty:
    st.warning("No customer data available.")
    st.stop()

# Recompute CLV dynamically based on user assumptions
survival_service = SurvivalAnalysisService.load()
clv_est = CLVEstimator(survival_service, default_margin=contribution_margin, default_discount_rate=discount_rate, default_horizon=horizon_months)

df_computed = clv_est.estimate_dataset_clv(
    df_clv_raw,
    churn_probabilities=df_clv_raw["churn_probability"].values,
    horizon_months=horizon_months,
    margin=contribution_margin,
    discount_rate=discount_rate,
)
merged = pd.merge(df_clv_raw, df_computed, on="customer_id")

# Aggregate KPIs
total_clv = float(merged["discounted_clv"].sum())
avg_clv = float(merged["discounted_clv"].mean())
total_rev_at_risk = float(merged["revenue_at_risk"].sum())
total_hist_rev = float(merged["total_charges"].sum())
future_gross_rev = float(merged["expected_future_revenue"].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Average Customer CLV", f"${avg_clv:,.2f}", f"{horizon_months}M Discounted Contribution")
with c2:
    render_metric_card("Portfolio Forward CLV", f"${total_clv:,.2f}", f"Gross Margin: {contribution_margin:.0%}")
with c3:
    render_metric_card("Total Revenue At Risk", f"${total_rev_at_risk:,.2f}", "Expected Churn Exposure", is_danger=True)
with c4:
    render_metric_card("Historical Realized Spend", f"${total_hist_rev:,.2f}", "Sum of Incurred Total Charges")

st.markdown("---")

tab_dist, tab_risk, tab_table = st.tabs(["📊 CLV Distribution & Value Segments", "🚨 Value vs Risk Matrix", "📋 Top High-Value Accounts"])

with tab_dist:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_hist = px.histogram(
            merged,
            x="discounted_clv",
            color="contract",
            nbins=40,
            title=f"Discounted CLV Distribution ({horizon_months} Months Horizon)",
            labels={"discounted_clv": "Discounted Customer Lifetime Value ($)", "contract": "Contract Type"},
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"],
        )
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        clv_by_contract = merged.groupby("contract")["discounted_clv"].mean().reset_index()
        fig_bar = px.bar(
            clv_by_contract,
            x="contract",
            y="discounted_clv",
            color="contract",
            text_auto=".2f",
            title="Average CLV by Contract Type ($)",
            labels={"discounted_clv": "Mean CLV ($)", "contract": "Contract"},
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"],
        )
        fig_bar.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab_risk:
    st.subheader("Revenue at Risk by Predictive Risk Tier")
    risk_summary = merged.groupby("risk_tier").agg(
        customer_count=("customer_id", "count"),
        avg_clv=("discounted_clv", "mean"),
        total_revenue_at_risk=("revenue_at_risk", "sum"),
    ).reset_index()

    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    risk_summary["order"] = risk_summary["risk_tier"].map(order).fillna(4)
    risk_summary = risk_summary.sort_values(by="order").drop(columns=["order"])
    
    fig_risk_bar = px.bar(
        risk_summary,
        x="risk_tier",
        y="total_revenue_at_risk",
        color="risk_tier",
        text_auto="$,.0f",
        title="Expected Revenue at Risk Across Churn Risk Tiers ($)",
        labels={"total_revenue_at_risk": "Total Revenue At Risk ($)", "risk_tier": "Risk Tier"},
        color_discrete_map={"Critical": "#991B1B", "High": "#C2410C", "Medium": "#B45309", "Low": "#166534"},
    )
    fig_risk_bar.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_risk_bar, use_container_width=True)

with tab_table:
    st.subheader("Top High-Value Accounts (Priority Retention Candidates)")
    top_accounts = merged.sort_values(by="discounted_clv", ascending=False).head(25)[
        ["customer_id", "contract", "tenure", "monthly_charges", "total_charges", "discounted_clv", "churn_probability", "revenue_at_risk", "risk_tier"]
    ]
    st.dataframe(top_accounts.style.format({
        "monthly_charges": "${:,.2f}",
        "total_charges": "${:,.2f}",
        "discounted_clv": "${:,.2f}",
        "revenue_at_risk": "${:,.2f}",
        "churn_probability": "{:.1%}",
    }), use_container_width=True, hide_index=True)

st.info(f"📘 {CLV_METHODOLOGY_DISCLAIMER}")
