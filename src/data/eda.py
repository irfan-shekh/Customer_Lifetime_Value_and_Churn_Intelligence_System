"""
Exploratory Data Analysis (EDA) module.
Calculates key business churn metrics, demographic segment comparisons,
and revenue at risk for telecom customers.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def calculate_eda_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary business metrics and rates across key dimensions.
    """
    total_customers = len(df)
    churn_count = int(df["churn"].sum()) if "churn" in df.columns else 0
    churn_rate = float(churn_count / total_customers) if total_customers > 0 else 0.0
    
    total_monthly_revenue = float(df["monthly_charges"].sum()) if "monthly_charges" in df.columns else 0.0
    churned_monthly_revenue = float(df[df["churn"] == 1]["monthly_charges"].sum()) if "churn" in df.columns else 0.0
    retained_monthly_revenue = total_monthly_revenue - churned_monthly_revenue
    
    total_historical_revenue = float(df["total_charges"].sum()) if "total_charges" in df.columns else 0.0
    churned_historical_revenue = float(df[df["churn"] == 1]["total_charges"].sum()) if "churn" in df.columns else 0.0

    # Churn rates by segment
    def get_group_churn(col: str) -> List[Dict[str, Any]]:
        if col not in df.columns or "churn" not in df.columns:
            return []
        grouped = df.groupby(col).agg(
            total_customers=("churn", "count"),
            churned_customers=("churn", "sum"),
            avg_monthly_charges=("monthly_charges", "mean"),
        ).reset_index()
        grouped["churn_rate"] = (grouped["churned_customers"] / grouped["total_customers"]).round(4)
        grouped["avg_monthly_charges"] = grouped["avg_monthly_charges"].round(2)
        return grouped.to_dict(orient="records")

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "retained_count": total_customers - churn_count,
        "churn_rate": round(churn_rate, 4),
        "total_monthly_revenue": round(total_monthly_revenue, 2),
        "churned_monthly_revenue": round(churned_monthly_revenue, 2),
        "retained_monthly_revenue": round(retained_monthly_revenue, 2),
        "monthly_revenue_at_risk_pct": round((churned_monthly_revenue / total_monthly_revenue * 100), 2) if total_monthly_revenue > 0 else 0.0,
        "total_historical_revenue": round(total_historical_revenue, 2),
        "churned_historical_revenue": round(churned_historical_revenue, 2),
        "churn_by_contract": get_group_churn("contract"),
        "churn_by_internet": get_group_churn("internet_service"),
        "churn_by_payment_method": get_group_churn("payment_method"),
        "churn_by_tenure_group": get_group_churn("tenure_group") if "tenure_group" in df.columns else [],
        "churn_by_senior": get_group_churn("senior_citizen"),
    }


def create_churn_by_contract_plot(df: pd.DataFrame) -> go.Figure:
    """Returns Plotly figure of churn rate by contract."""
    rates = df.groupby("contract")["churn"].mean().reset_index()
    rates["churn_pct"] = (rates["churn"] * 100).round(1)
    
    fig = px.bar(
        rates,
        x="contract",
        y="churn_pct",
        color="contract",
        text="churn_pct",
        title="Churn Rate by Contract Type",
        labels={"churn_pct": "Churn Rate (%)", "contract": "Contract Type"},
        color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 60], template="plotly_white", showlegend=False)
    return fig


def create_tenure_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Returns Plotly histogram of tenure split by churn status."""
    fig = px.histogram(
        df,
        x="tenure",
        color=df["churn"].map({1: "Churned", 0: "Retained"}),
        barmode="overlay",
        nbins=36,
        title="Customer Tenure Distribution (Months)",
        labels={"tenure": "Tenure (Months)", "color": "Status"},
        color_discrete_map={"Churned": "#EF4444", "Retained": "#3B82F6"},
        opacity=0.75,
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_monthly_charges_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Returns Plotly KDE/Box plot of monthly charges by churn status."""
    fig = px.box(
        df,
        x=df["churn"].map({1: "Churned", 0: "Retained"}),
        y="monthly_charges",
        color=df["churn"].map({1: "Churned", 0: "Retained"}),
        title="Monthly Charges by Churn Status ($)",
        labels={"x": "Customer Status", "monthly_charges": "Monthly Charges ($)", "color": "Status"},
        color_discrete_map={"Churned": "#EF4444", "Retained": "#3B82F6"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig
