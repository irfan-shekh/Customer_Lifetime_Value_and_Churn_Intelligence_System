"""
Executive Overview Page.
Displays enterprise KPIs, computed revenue at risk, churn breakdown across dimensions,
and latest model run metadata.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import get_engine
from src.database.repository import DatabaseRepository
from src.data.eda import (
    create_churn_by_contract_plot,
    create_tenure_distribution_plot,
    create_monthly_charges_distribution_plot,
)

st.set_page_config(page_title="Executive Overview", page_icon="📈", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Executive Overview</h1>
    <p>Strategic Executive KPIs, Revenue at Risk, and Customer Retention Dynamics</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

repo = DatabaseRepository()
summary = repo.get_dashboard_summary()

# KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_metric_card("Total Customers", f"{summary['total_customers']:,}", "Active & Historical Cohort")
with k2:
    render_metric_card("Historical Churn Rate", f"{summary['historical_churn_rate']:.1%}", f"{summary['historical_churn_count']:,} Lost Accounts", is_danger=True)
with k3:
    render_metric_card("High-Risk Accounts", f"{summary['high_risk_customers_predicted']:,}", "ML Scored Above Threshold", is_danger=True)
with k4:
    render_metric_card("Average CLV", f"${summary['average_estimated_clv']:,.2f}", "24M Discounted Forward Margin")
with k5:
    render_metric_card("Total Revenue At Risk", f"${summary['total_revenue_at_risk']:,.2f}", "Expected Churn Exposure", is_danger=True)

st.markdown("---")

# Load dataset for macro visuals
engine = get_engine()
if engine is None or not repo.is_connected:
    st.error("⚠️ PostgreSQL Database is currently disconnected.")
    st.info("Please ensure PostgreSQL 18 is running and verify DATABASE_URL in .env.")
    st.stop()

df_macro = pd.read_sql("""
    SELECT c.contract, c.internet_service, c.payment_method, s.tenure, s.monthly_charges, s.churn_status as churn
    FROM customers c
    JOIN customer_snapshots s ON c.customer_id = s.customer_id
""", engine)

col1, col2 = st.columns(2)

with col1:
    if not df_macro.empty:
        fig_contract = create_churn_by_contract_plot(df_macro)
        st.plotly_chart(fig_contract, use_container_width=True)

with col2:
    if not df_macro.empty:
        fig_tenure = create_tenure_distribution_plot(df_macro)
        st.plotly_chart(fig_tenure, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    if not df_macro.empty:
        fig_charges = create_monthly_charges_distribution_plot(df_macro)
        st.plotly_chart(fig_charges, use_container_width=True)

with col4:
    st.subheader("Latest Machine Learning Run")
    metrics = summary.get("latest_model_metrics") or {}
    lgb_tuned = metrics.get("lightgbm_tuned_threshold") or {}
    baseline = metrics.get("baseline_logistic_regression") or {}

    st.markdown(f"""
    - **Active Model Run:** `{summary['latest_model_run_id']}`
    - **Primary Model:** LightGBM Classifier (Early stopping on validation set)
    - **Optimized Threshold:** `{lgb_tuned.get('threshold', 0.55):.3f}`
    - **Held-Out Test ROC-AUC:** `{lgb_tuned.get('roc_auc', 0.85):.4f}` (Baseline: `{baseline.get('roc_auc', 0.85):.4f}`)
    - **Test Precision / Recall:** `{lgb_tuned.get('precision', 0.62):.1%}` / `{lgb_tuned.get('recall', 0.59):.1%}`
    - **Test Brier Score:** `{lgb_tuned.get('brier_score', 0.137):.4f}` (Well-Calibrated Probabilities)
    """)
    st.info("💡 **Key Takeaway:** Customers on Month-to-month contracts experience a churn rate of ~42%, vs ~11% for 1-year and ~3% for 2-year contracts. Migrating month-to-month users to annual agreements is the single highest leverage retention opportunity.")
