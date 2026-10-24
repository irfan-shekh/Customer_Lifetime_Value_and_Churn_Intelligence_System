"""
Retention Campaigns & Uplift Modeling Page.
Demonstrates causal uplift estimation, identifies 'Persuadables', evaluates Qini curves,
and articulates observational causal limitations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import get_engine
from src.models.uplift import (
    generate_demonstration_rct_dataset,
    TLearnerUpliftModel,
    UPLIFT_METHODOLOGY_DISCLAIMER,
)
from src.evaluation.uplift import calculate_qini_curve, create_qini_plot

st.set_page_config(page_title="Retention Campaigns", page_icon="🎯", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Retention Campaigns & Uplift Modeling</h1>
    <p>Heterogeneous Treatment Effects, Persuadables Targeting, Qini Curves & Randomized Trial Design</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

# Methodological Alert
st.error(f"🚨 {UPLIFT_METHODOLOGY_DISCLAIMER}")

# Load active campaigns from database
engine = get_engine()
if engine is None:
    st.error("⚠️ PostgreSQL Database is currently disconnected.")
    st.info("Please ensure PostgreSQL 18 is running and verify DATABASE_URL in .env.")
    st.stop()

df_camps = pd.read_sql("SELECT * FROM retention_campaigns ORDER BY roi_pct DESC", engine)

st.subheader("📋 Active Campaign Scenarios in Database")
if not df_camps.empty:
    st.dataframe(df_camps[[
        "campaign_name", "target_risk_tier", "discount_pct", "discount_duration_months",
        "eligible_customers", "estimated_retained_customers", "campaign_cost",
        "retained_contribution", "net_benefit", "roi_pct"
    ]].style.format({
        "discount_pct": "{:.0%}",
        "eligible_customers": "{:,}",
        "estimated_retained_customers": "{:,}",
        "campaign_cost": "${:,.2f}",
        "retained_contribution": "${:,.2f}",
        "net_benefit": "${:,.2f}",
        "roi_pct": "{:.1f}%",
    }), use_container_width=True, hide_index=True)

st.markdown("---")

st.subheader("🧪 Uplift Modeling: The Persuadables Quadrant")
st.markdown("""
Traditional churn prediction targets customers with the highest **P(Churn)**.
However, this leads to two major forms of marketing waste:
1. **Sure Things:** Customers who would have stayed anyway (wasting discount budget).
2. **Lost Causes:** Customers who will churn regardless of offer (burning campaign spend with zero retention).
3. **Sleeping Dogs:** Customers for whom receiving marketing outreach triggers cancellation.

**Uplift Modeling** isolates the **Persuadables**—customers who only remain active *because* they received the intervention.
""")

col_quad1, col_quad2 = st.columns([1, 1])

# Run lightweight synthetic RCT demonstration for visual proof
@st.cache_data
def get_demonstration_uplift_data():
    df_raw = pd.read_sql("SELECT * FROM customers c JOIN customer_snapshots s ON c.customer_id = s.customer_id LIMIT 2500", engine)
    rct_df = generate_demonstration_rct_dataset(df_raw)
    
    # Simple feature matrix
    X_cols = ["tenure", "monthly_charges"]
    rct_df["is_m2m"] = (rct_df["contract"].astype(str).str.lower() == "month-to-month").astype(int)
    X = rct_df[["tenure", "monthly_charges", "is_m2m"]]
    
    model = TLearnerUpliftModel()
    model.fit(X, rct_df["treatment_offer"].values, rct_df["synthetic_rct_churn_outcome"].values)
    uplift_scores = model.predict_uplift(X)
    
    qini_res = calculate_qini_curve(
        treatment=rct_df["treatment_offer"].values,
        outcome=rct_df["synthetic_rct_churn_outcome"].values,
        uplift_scores=uplift_scores,
    )
    
    rct_df["predicted_uplift"] = uplift_scores
    quadrants = model.categorize_customers(
        X,
        baseline_churn_prob=rct_df["synthetic_rct_churn_outcome"].values,
        uplift_scores=uplift_scores,
    )
    rct_df["uplift_quadrant"] = quadrants
    return rct_df, qini_res

rct_df, qini_res = get_demonstration_uplift_data()

with col_quad1:
    quad_counts = rct_df["uplift_quadrant"].value_counts().reset_index()
    fig_pie = px.pie(
        quad_counts,
        names="uplift_quadrant",
        values="count",
        title="Simulated Customer Uplift Quadrant Breakdown",
        color="uplift_quadrant",
        color_discrete_map={
            "Persuadables (Target)": "#10B981",
            "Sure Things (Organic Stayers)": "#3B82F6",
            "Lost Causes (Low Response)": "#94A3B8",
            "Sleeping Dogs (Do Not Disturb)": "#EF4444",
        },
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_quad2:
    fig_qini = create_qini_plot(qini_res)
    st.plotly_chart(fig_qini, use_container_width=True)

st.markdown("""
### 📋 How to Implement Causal Validation in Production
To validate true incremental treatment effects, the telecom business must execute an **A/B Test**:
1. Select target population (e.g. Month-to-month subscribers with high CLV).
2. Randomly assign 90% to Treatment (receive retention discount) and 10% to Holdout Control (no discount).
3. Measure 90-day churn outcomes in both groups.
4. Fit Meta-Learners (`CausalML` or `TLearner`) on realized experimental data to extract true causal uplift $\tau(X)$.
""")
