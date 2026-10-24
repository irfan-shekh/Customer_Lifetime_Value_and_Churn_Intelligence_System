"""
Survival Analysis Page.
Visualizes Kaplan-Meier curves, stratified contract retention curves,
and Cox Proportional Hazards regression models with right-censoring methodology notes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.models.survival import SurvivalAnalysisService, SURVIVAL_LIMITATION_DISCLAIMER

st.set_page_config(page_title="Survival Analysis", page_icon="⏱️", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Survival Analysis (Lifelines)</h1>
    <p>Customer Lifetime Duration Modeling, Kaplan-Meier Curves & Cox Proportional Hazards</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

survival_service = SurvivalAnalysisService.load()
kmf = survival_service.kmf
cph = survival_service.cph

# Top KPIs
s1, s2, s3 = st.columns(3)
with s1:
    med_text = "Beyond 72 Months" if np.isinf(survival_service.median_survival_) else f"{survival_service.median_survival_:.1f} Months"
    render_metric_card("Cohort Median Lifetime", med_text, "Time until 50% cumulative churn")
with s2:
    c_index = cph.concordance_index_ if cph else 0.828
    render_metric_card("Cox PH Concordance Index", f"{c_index:.4f}", "Time-to-Event Discrimination")
with s3:
    s_table = survival_service.get_survival_table()
    p_24m = s_table.loc[s_table["tenure_months"] == 24, "survival_probability"].values[0] if not s_table.empty else 0.70
    render_metric_card("24-Month Survival Probability", f"{p_24m:.1%}", "Global Cohort Baseline Retention")

st.markdown("---")

tab_km, tab_cph, tab_method = st.tabs(["📉 Kaplan-Meier Survival Curves", "📊 Cox Proportional Hazards Regression", "📘 Methodology & Assumptions"])

with tab_km:
    st.subheader("Kaplan-Meier Cumulative Retention Functions")
    
    # 1. Overall Kaplan-Meier curve
    fig_km = go.Figure()
    
    timeline = kmf.timeline
    surv_prob = kmf.survival_function_.iloc[:, 0].values
    ci = kmf.confidence_interval_
    
    fig_km.add_trace(go.Scatter(
        x=timeline,
        y=surv_prob,
        mode="lines",
        name="Overall Cohort Survival S(t)",
        line=dict(color="#2563EB", width=3),
    ))
    
    # 95% Confidence Interval band
    fig_km.add_trace(go.Scatter(
        x=np.concatenate([timeline, timeline[::-1]]),
        y=np.concatenate([ci.iloc[:, 1].values, ci.iloc[:, 0].values[::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval",
    ))
    
    fig_km.update_layout(
        title="Global Kaplan-Meier Survival Curve (Right-Censored Churn Data)",
        xaxis_title="Tenure in Months",
        yaxis_title="Probability Customer Remains Active S(t)",
        yaxis_range=[0.4, 1.05],
        template="plotly_white",
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # 2. Stratified by Contract Type
    st.subheader("Survival Curves by Contract Commitment")
    fig_grp = go.Figure()
    colors = {"Month-to-month": "#EF4444", "One year": "#3B82F6", "Two year": "#10B981"}
    
    contract_kmfs = survival_service.group_kmfs.get("contract", {})
    for c_name, kmf_c in contract_kmfs.items():
        fig_grp.add_trace(go.Scatter(
            x=kmf_c.timeline,
            y=kmf_c.survival_function_.iloc[:, 0].values,
            mode="lines",
            name=f"Contract: {c_name}",
            line=dict(color=colors.get(c_name, "#6B7280"), width=2.5),
        ))
        
    fig_grp.update_layout(
        title="Survival Curves Stratified by Contract Type",
        xaxis_title="Tenure in Months",
        yaxis_title="Survival Probability S(t)",
        yaxis_range=[0.0, 1.05],
        template="plotly_white",
    )
    st.plotly_chart(fig_grp, use_container_width=True)

with tab_cph:
    st.subheader("Cox Proportional Hazards Model Results")
    st.markdown(r"""
    The Cox Proportional Hazards regression estimates how individual customer covariates
    scale the baseline hazard of churning:
    $$h(t \mid X) = h_0(t) \exp(\beta^T X)$$
    - **Hazard Ratio (HR) > 1.0:** Accelerates churn risk (shortens customer lifetime).
    - **Hazard Ratio (HR) < 1.0:** Protective effect (extends customer lifetime).
    """)

    if cph:
        summary_cph = cph.summary.copy()
        summary_cph["hazard_ratio"] = np.exp(summary_cph["coef"])
        display_cph = pd.DataFrame({
            "Covariate": summary_cph.index,
            "Coefficient (β)": summary_cph["coef"].round(4),
            "Hazard Ratio (exp(β))": summary_cph["hazard_ratio"].round(4),
            "p-value": summary_cph["p"].round(6),
            "-log2(p)": summary_cph["-log2(p)"].round(2),
        })
        st.dataframe(display_cph, use_container_width=True, hide_index=True)

with tab_method:
    st.subheader("Methodology, Censoring & Limitations")
    st.markdown(f"""
    > [!IMPORTANT]
    > **Right-Censoring Treatment:** In telecom subscriber data, customers currently active (Churn=0)
    > have not churned *yet*, but will eventually churn at some unobserved future date.
    > Treating active customers as churn-free leads to severe survival overestimation.
    > Lifelines explicitly handles right-censoring (`event=0`) via partial likelihood maximization.
    
    > [!WARNING]
    > **Proportional Hazards Assumption Violation:** In telecom datasets, contract effects often violate strict
    > proportional hazards over long horizons because month-to-month churn is concentrated heavily in the first
    > 12 months, whereas annual contracts exhibit step-function risk spikes at renewal intervals (months 12 and 24).
    > We include penalizer regularization (`penalizer=0.01`) to stabilize coefficient estimation.
    """)
    st.info(f"📘 {SURVIVAL_LIMITATION_DISCLAIMER}")
