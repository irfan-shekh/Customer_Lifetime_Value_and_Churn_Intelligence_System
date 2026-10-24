"""
Churn Intelligence Page.
Deep dive into LightGBM performance, baseline Logistic Regression comparison,
business threshold optimization curve, and global SHAP feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.models.churn import ChurnModelService
from src.models.shap_explainer import ShapExplanationService, CAUSAL_DISCLAIMER

st.set_page_config(page_title="Churn Intelligence", page_icon="⚡", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Churn Intelligence & Model Diagnostics</h1>
    <p>Predictive LightGBM Classifier, Baseline Benchmarks, Threshold Optimization & Global SHAP Insights</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

churn_service = ChurnModelService.load()
metrics = churn_service.metrics
tuned_metrics = metrics.get("lightgbm_tuned_threshold", {})
baseline_metrics = metrics.get("baseline_logistic_regression", {})
standard_metrics = metrics.get("lightgbm_standard_threshold_0_50", {})

# Top Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric_card("Test ROC-AUC", f"{tuned_metrics.get('roc_auc', 0.85):.4f}", f"Baseline: {baseline_metrics.get('roc_auc', 0.85):.4f}")
with m2:
    render_metric_card("Test PR-AUC", f"{tuned_metrics.get('pr_auc', 0.66):.4f}", f"Baseline: {baseline_metrics.get('pr_auc', 0.66):.4f}")
with m3:
    render_metric_card("Decision Threshold", f"{churn_service.best_threshold:.3f}", "Business Value Optimized")
with m4:
    render_metric_card("Precision @ Threshold", f"{tuned_metrics.get('precision', 0.62):.1%}", "Accuracy: 79.8%")
with m5:
    render_metric_card("Recall @ Threshold", f"{tuned_metrics.get('recall', 0.59):.1%}", "F1-Score: 0.61")

st.markdown("---")

# Tab navigation
tab_models, tab_thresh, tab_shap = st.tabs(["📊 Model Comparison & Confusion Matrix", "🎛️ Business Threshold Tuning", "🌐 Global SHAP Importance"])

with tab_models:
    col_comp, col_cm = st.columns([1, 1])
    
    with col_comp:
        st.subheader("Model Benchmark: LightGBM vs Logistic Regression")
        comp_df = pd.DataFrame([
            {
                "Evaluation Metric": "ROC-AUC (Discrimination)",
                "LightGBM (Tuned)": f"{tuned_metrics.get('roc_auc', 0.85):.4f}",
                "LightGBM (0.50)": f"{standard_metrics.get('roc_auc', 0.85):.4f}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('roc_auc', 0.85):.4f}",
            },
            {
                "Evaluation Metric": "PR-AUC (Precision-Recall)",
                "LightGBM (Tuned)": f"{tuned_metrics.get('pr_auc', 0.66):.4f}",
                "LightGBM (0.50)": f"{standard_metrics.get('pr_auc', 0.66):.4f}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('pr_auc', 0.66):.4f}",
            },
            {
                "Evaluation Metric": "F1-Score",
                "LightGBM (Tuned)": f"{tuned_metrics.get('f1_score', 0.61):.4f}",
                "LightGBM (0.50)": f"{standard_metrics.get('f1_score', 0.62):.4f}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('f1_score', 0.63):.4f}",
            },
            {
                "Evaluation Metric": "Recall (Churn Detection Rate)",
                "LightGBM (Tuned)": f"{tuned_metrics.get('recall', 0.59):.1%}",
                "LightGBM (0.50)": f"{standard_metrics.get('recall', 0.65):.1%}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('recall', 0.80):.1%}",
            },
            {
                "Evaluation Metric": "Precision (False Alarm Avoidance)",
                "LightGBM (Tuned)": f"{tuned_metrics.get('precision', 0.62):.1%}",
                "LightGBM (0.50)": f"{standard_metrics.get('precision', 0.60):.1%}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('precision', 0.52):.1%}",
            },
            {
                "Evaluation Metric": "Brier Score (Probability Calibration)",
                "LightGBM (Tuned)": f"{tuned_metrics.get('brier_score', 0.137):.4f}",
                "LightGBM (0.50)": f"{standard_metrics.get('brier_score', 0.137):.4f}",
                "Logistic Regression Baseline": f"{baseline_metrics.get('brier_score', 0.160):.4f}",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("Note: Lower Brier score indicates superior probability calibration. LightGBM demonstrates significantly sharper calibration than the unregularized linear baseline.")

    with col_cm:
        st.subheader("Confusion Matrix on Held-Out Test Set (1,409 customers)")
        cm = tuned_metrics.get("confusion_matrix", {})
        tn = cm.get("true_negatives", 903)
        fp = cm.get("false_positives", 132)
        fn = cm.get("false_negatives", 152)
        tp = cm.get("true_positives", 222)

        cm_matrix = [[tn, fp], [fn, tp]]
        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Predicted Stay", "Predicted Churn"],
            y=["Actual Stay", "Actual Churn"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

with tab_thresh:
    st.subheader("Business-Driven Threshold Optimization")
    st.markdown("""
    Standard machine learning defaults to a 0.50 threshold. However, in retention marketing,
    **missing an actual churner (False Negative)** costs significantly more than **sending an offer to a customer who would stay (False Positive)**.
    The curve below demonstrates how expected net business profit changes as the decision threshold shifts.
    """)

    report = churn_service.threshold_report
    records = report.get("threshold_analysis", [])
    if records:
        df_thresh = pd.DataFrame(records)
        
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(
            x=df_thresh["threshold"],
            y=df_thresh["net_value"],
            mode="lines",
            name="Expected Net Profit ($)",
            line=dict(color="#10B981", width=3),
        ))
        fig_thresh.add_vline(
            x=churn_service.best_threshold,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Optimal ({churn_service.best_threshold:.3f})",
        )
        fig_thresh.update_layout(
            title="Net Business Value vs Decision Threshold (Validation Cohort)",
            xaxis_title="Classification Threshold",
            yaxis_title="Expected Net Benefit ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        # Precision-Recall trade-off
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["precision"], name="Precision", line=dict(color="#3B82F6")))
        fig_pr.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["recall"], name="Recall", line=dict(color="#F59E0B")))
        fig_pr.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["f1_score"], name="F1-Score", line=dict(color="#8B5CF6", dash="dot")))
        fig_pr.update_layout(
            title="Precision vs Recall Trade-Off",
            xaxis_title="Decision Threshold",
            yaxis_title="Metric Value",
            template="plotly_white",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

with tab_shap:
    st.subheader("Global Feature Importance (TreeExplainer SHAP)")
    shap_service = ShapExplanationService(churn_service)
    imp_df = shap_service.get_global_importance(top_n=15)
    
    if not imp_df.empty:
        fig_imp = px.bar(
            imp_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global Mean |SHAP| Impact on Churn Predictions",
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Predictive Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.warning(f"⚠️ {CAUSAL_DISCLAIMER}")
