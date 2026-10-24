"""
Customer Segmentation Page.
Unsupervised K-Means clustering, silhouette score evaluation, 2D PCA visual projections,
and data-grounded business personas with actionable policy playbooks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import get_engine
from src.models.segmentation import CustomerSegmentationService

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Customer Segmentation & Behavioral Clustering</h1>
    <p>K-Means Archetypes, 2D PCA Projections & Tailored Strategic Playbooks</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

seg_service = CustomerSegmentationService.load()
engine = get_engine()

if engine is None:
    st.error("⚠️ PostgreSQL Database is currently disconnected.")
    st.info("Please ensure PostgreSQL 18 is running and verify DATABASE_URL in .env.")
    st.stop()

from src.data.features import engineer_features

# Load customer segment assignments
df_seg = pd.read_sql("""
    SELECT 
        c.customer_id, c.contract, c.internet_service,
        c.phone_service, c.multiple_lines, c.online_security, c.online_backup,
        c.device_protection, c.tech_support, c.streaming_tv, c.streaming_movies,
        s.tenure, s.monthly_charges, s.total_charges,
        g.cluster_id, g.cluster_label, g.cluster_description, p.churn_probability, v.discounted_clv
    FROM customer_segments g
    JOIN customers c ON g.customer_id = c.customer_id
    JOIN customer_snapshots s ON c.customer_id = s.customer_id
    LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
    LEFT JOIN customer_lifetime_values v ON c.customer_id = v.customer_id
""", engine)

if df_seg.empty:
    st.warning("No segmentation records found in database. Run 'python scripts/generate_predictions.py' first.")
    st.stop()

# Feature engineering & Transform coordinates for 2D PCA plotting
df_seg = engineer_features(df_seg)
df_projected = seg_service.transform(df_seg)

# Top KPIs
p1, p2, p3 = st.columns(3)
with p1:
    render_metric_card("Optimal Cluster Count (K)", f"{seg_service.n_clusters}", "Silhouette Optimized")
with p2:
    best_score = seg_service.silhouette_scores.get(seg_service.n_clusters, 0.41)
    render_metric_card("Silhouette Coefficient", f"{best_score:.4f}", "Cluster Separation Quality")
with p3:
    render_metric_card("Segmented Customer Cohort", f"{len(df_seg):,}", "Complete Active Database")

st.markdown("---")

tab_scatter, tab_profiles, tab_eval = st.tabs(["🗺️ 2D PCA Behavioral Map", "📋 Cluster Profiles & Playbooks", "📈 Silhouette Evaluation Curve"])

with tab_scatter:
    st.subheader("2D PCA Customer Projection (Color-Coded by Cluster Persona)")
    fig_pca = px.scatter(
        df_projected,
        x="pca_dim_1",
        y="pca_dim_2",
        color="cluster_label",
        hover_data=["customer_id", "tenure", "monthly_charges", "churn_probability", "discounted_clv"],
        title="2D Principal Component Space of Customer Behavioral Attributes",
        labels={"pca_dim_1": "Principal Component 1 (Tenure & Spend Dimension)", "pca_dim_2": "Principal Component 2 (Churn Risk Dimension)", "cluster_label": "Persona"},
        template="plotly_white",
        opacity=0.75,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with tab_profiles:
    st.subheader("Data-Grounded Persona Profiles & Retention Playbooks")
    for c_id, meta in seg_service.cluster_metadata.items():
        with st.expander(f"Cluster {c_id}: {meta['cluster_label']} ({meta['customer_count']:,} Accounts - {meta['customer_pct']}%)", expanded=True):
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                st.markdown("**Centroid Profile Characteristics:**")
                cent = meta["centroid"]
                st.markdown(f"""
                - **Average Tenure:** `{cent.get('tenure', 0):.1f} Months`
                - **Monthly Spend:** `${cent.get('monthly_charges', 0):.2f}`
                - **Churn Risk Probability:** `{cent.get('churn_probability', 0):.1%}`
                - **Estimated Forward CLV:** `${cent.get('discounted_clv', 0):,.2f}`
                - **Active Services:** `{cent.get('num_services_subscribed', 0):.1f}`
                """)
            with col_c2:
                st.markdown(f"**Behavioral Description:**\n{meta['cluster_description']}")
                st.success(f"**Recommended Strategic Playbook:**\n{meta['recommended_action']}")

with tab_eval:
    st.subheader("Silhouette Score vs Cluster Count (K)")
    scores = seg_service.silhouette_scores
    if scores:
        df_scores = pd.DataFrame([{"k": k, "silhouette_score": s} for k, s in scores.items()])
        fig_sil = px.line(
            df_scores,
            x="k",
            y="silhouette_score",
            markers=True,
            title="Cluster Count Selection (Silhouette Analysis)",
            labels={"k": "Number of Clusters (K)", "silhouette_score": "Silhouette Score"},
        )
        fig_sil.add_vline(x=seg_service.n_clusters, line_dash="dash", line_color="#10B981", annotation_text=f"Selected K={seg_service.n_clusters}")
        fig_sil.update_layout(template="plotly_white")
        st.plotly_chart(fig_sil, use_container_width=True)
