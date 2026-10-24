"""
Customer Lifetime Value and Churn Intelligence System.
Main Streamlit Application Entrypoint.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.ui_components import inject_custom_css, render_storage_status_banner
from src.database.repository import DatabaseRepository
from src.database.connection import check_connection_health

st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Header
st.markdown("""
<div class="header-banner">
    <h1>Customer Lifetime Value & Churn Intelligence System</h1>
    <p>Enterprise Machine Learning, Survival Dynamics, Causal Uplift & Financial Decision Engine</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

# Quick Overview & Navigation Guide
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("System Objective & Business Architecture")
    st.markdown("""
    Telecom and banking organizations face substantial recurring revenue losses when customers churn.
    This system connects advanced statistical learning to rigorous financial decision-making:
    
    1. **Predictive Risk Modeling (LightGBM):** Calibrated customer-level churn risk estimation with validation-tuned decision thresholds.
    2. **Explainable AI (SHAP):** Granular global and individual feature attributions separating predictive drivers from causal claims.
    3. **Survival Dynamics (Lifelines):** Kaplan-Meier and Cox Proportional Hazards modeling to estimate retention over 12, 24, 36, and 48-month horizons.
    4. **Forward-Looking CLV:** Actuarial discounted cash flow estimation isolating historical charges from future margin.
    5. **Behavioral Segmentation:** K-Means clustering and PCA dimensionality reduction delivering data-grounded business personas.
    6. **Causal Uplift & ROI Simulator:** Heterogeneous treatment effect estimation identifying *Persuadables* and calculating campaign ROI.
    7. **Enterprise Persistence:** Normalized PostgreSQL database layer with safe upserts and health telemetry.
    """)

with col2:
    st.subheader("Quick System Status")
    db_health = check_connection_health()
    repo = DatabaseRepository()
    summary = repo.get_dashboard_summary()

    if db_health["status"] == "HEALTHY":
        st.info(f"**Storage Engine:** {db_health['storage_mode']}")
        st.metric("Total Ingested Customers", f"{summary['total_customers']:,}")
        st.metric("High-Risk Customers", f"{summary['high_risk_customers_predicted']:,}")
        st.metric("Average Estimated CLV", f"${summary['average_estimated_clv']:,.2f}")
        st.caption(f"Latest Model Run: `{summary['latest_model_run_id']}`")
    else:
        st.error(f"**Storage Engine:** {db_health['storage_mode']} ({db_health['status']})")
        st.warning(f"Database Message: {db_health['message']}")
        st.info("Tip: If running Streamlit from the command line, ensure your virtual environment is active: `.\\.venv\\Scripts\\Activate.ps1`")

st.markdown("---")

st.subheader("Dashboard Modules")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    #### 📈 Strategy & Monitoring
    - **1_Executive_Overview:** Executive KPIs, revenue at risk, and macro distributions.
    - **2_Customer_Explorer:** 360-degree customer search, SHAP attribution, and prescriptive actions.
    - **3_Churn_Intelligence:** LightGBM vs baseline, threshold optimization, and confusion matrices.
    """)

with c2:
    st.markdown("""
    #### ⏱️ Valuation & Segmentation
    - **4_Survival_Analysis:** Kaplan-Meier survival curves, Cox PH regression, and median lifetimes.
    - **5_Customer_Lifetime_Value:** Survival-discounted future margin and revenue at risk.
    - **6_Customer_Segmentation:** K-Means clustering, silhouette optimization, and 2D PCA profiles.
    """)

with c3:
    st.markdown("""
    #### 🎯 Retention Campaigns & Governance
    - **7_Retention_Campaigns:** Uplift modeling, Persuadables vs Lost Causes, and Qini curves.
    - **8_ROI_Simulator:** Interactive campaign financial calculator and scenario comparisons.
    - **9_Model_Evaluation:** Diagnostics, slice-based demographic evaluations, and leakage audit.
    - **10_Data_Database_Health:** Live PostgreSQL telemetry, schema dictionary, and connection health.
    """)
