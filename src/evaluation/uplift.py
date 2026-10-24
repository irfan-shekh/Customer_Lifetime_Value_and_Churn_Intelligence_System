"""
Uplift Model Evaluation Module.
Computes Qini curves, Cumulative Uplift curves, Area Under the Uplift Curve (AUUC),
and Qini coefficient.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def calculate_qini_curve(
    treatment: np.ndarray,
    outcome: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Qini curve and Area Under Qini Curve.
    Qini Curve = n_t,1 - (n_c,1 * N_t / N_c)
    Where:
      - n_t,1 = retained in treatment (or churned prevented)
      - n_c,1 = retained in control
    """
    df = pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,  # 1 if churned, 0 if retained
        "uplift": uplift_scores,
    })
    
    # Sort descending by predicted uplift (targeting best customers first)
    df = df.sort_values(by="uplift", ascending=False).reset_index(drop=True)

    # Invert outcome so 1 = Retained, 0 = Churned for positive response
    df["retained"] = 1 - df["outcome"]

    total_t = int((df["treatment"] == 1).sum())
    total_c = int((df["treatment"] == 0).sum())
    ratio_tc = (total_t / total_c) if total_c > 0 else 1.0

    df["cum_t_retained"] = (df["treatment"] * df["retained"]).cumsum()
    df["cum_c_retained"] = ((1 - df["treatment"]) * df["retained"]).cumsum()

    # Cumulative Qini value
    df["qini_value"] = df["cum_t_retained"] - (df["cum_c_retained"] * ratio_tc)
    df["population_pct"] = np.linspace(0.0, 100.0, len(df))

    # Downsample points for dashboard chart
    step = max(1, len(df) // n_bins)
    sampled = df.iloc[::step].copy()
    
    # Random baseline line connects (0,0) to final Qini value
    final_qini = float(df["qini_value"].iloc[-1])
    sampled["random_baseline"] = np.linspace(0.0, final_qini, len(sampled))

    # Calculate approximate Area Under Qini Curve (trapezoidal rule)
    auuc = float(np.trapz(df["qini_value"], df["population_pct"] / 100.0))
    random_area = 0.5 * final_qini
    qini_score = auuc - random_area

    return {
        "auuc": round(auuc, 2),
        "qini_score": round(qini_score, 2),
        "population_pct": sampled["population_pct"].round(1).tolist(),
        "qini_curve": sampled["qini_value"].round(1).tolist(),
        "random_baseline": sampled["random_baseline"].round(1).tolist(),
    }


def create_qini_plot(qini_data: Dict[str, Any]) -> go.Figure:
    """Generates Plotly figure for Qini Cumulative Uplift Curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=qini_data["population_pct"],
        y=qini_data["qini_curve"],
        mode="lines+markers",
        name="Uplift Model (T-Learner)",
        line=dict(color="#10B981", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=qini_data["population_pct"],
        y=qini_data["random_baseline"],
        mode="lines",
        name="Random Targeting Baseline",
        line=dict(color="#9CA3AF", dash="dash", width=2),
    ))

    fig.update_layout(
        title=f"Cumulative Uplift Qini Curve (Qini Score: +{qini_data['qini_score']})",
        xaxis_title="Percentage of Customers Targeted (%)",
        yaxis_title="Cumulative Incremental Customers Saved",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
