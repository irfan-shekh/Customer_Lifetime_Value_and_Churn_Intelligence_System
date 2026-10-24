"""
Customer Lifetime Value (CLV) Module.
Computes actuarially sound, survival-discounted expected future cash flows and revenue at risk.
Uses high-performance vectorized lookup tables over unique tenure values.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from src.config import (
    DEFAULT_CONTRIBUTION_MARGIN,
    DEFAULT_ANNUAL_DISCOUNT_RATE,
    DEFAULT_HORIZON_MONTHS,
    MODEL_ARTIFACTS_DIR,
)
from src.models.survival import SurvivalAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLV_METHODOLOGY_DISCLAIMER = (
    "Valuation Notice: Customer Lifetime Value (CLV) is calculated as the sum of discounted expected "
    "future contribution margins over a specified forecast horizon, weighted by monthly conditional survival probabilities. "
    "It assumes constant monthly spend and does not account for future contractual price revisions or macroeconomic shocks. "
    "Historical TotalCharges reflects realized revenue to date and is strictly distinguished from forward-looking CLV."
)


class CLVEstimator:
    """Calculates forward-looking customer lifetime values using survival retention projections."""

    def __init__(
        self,
        survival_service: Optional[SurvivalAnalysisService] = None,
        default_margin: float = DEFAULT_CONTRIBUTION_MARGIN,
        default_discount_rate: float = DEFAULT_ANNUAL_DISCOUNT_RATE,
        default_horizon: int = DEFAULT_HORIZON_MONTHS,
    ):
        self.survival_service = survival_service or SurvivalAnalysisService.load()
        self.default_margin = default_margin
        self.default_discount_rate = default_discount_rate
        self.default_horizon = default_horizon
        self._tenure_cache: Dict[Tuple[int, float, float], Dict[int, Tuple[float, float]]] = {}

    def _get_multipliers_table(
        self,
        horizon_months: int,
        margin: float,
        discount_rate: float,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Precomputes (rev_multiplier, clv_multiplier) for all possible tenures t0 in range(0, 100).
        Rev Multiplier = sum_{m=1}^H [S(t0 + m) / S(t0)]
        CLV Multiplier = sum_{m=1}^H [S(t0 + m) / S(t0) * margin * discount_factor^m]
        """
        key = (horizon_months, margin, discount_rate)
        if key in self._tenure_cache:
            return self._tenure_cache[key]

        kmf = self.survival_service.kmf
        monthly_df = (1.0 + discount_rate) ** (-1.0 / 12.0)
        table = {}

        # Extract survival function as an array/series once
        timeline = np.arange(0, 120)
        s_vals = kmf.survival_function_at_times(timeline).values if kmf else np.ones(len(timeline))

        for t0 in range(0, 100):
            t0_clean = max(1, t0)
            s_t0 = s_vals[t0_clean] if t0_clean < len(s_vals) else s_vals[-1]
            if s_t0 <= 1e-6:
                table[t0] = (0.0, 0.0)
                continue

            rev_mult = 0.0
            clv_mult = 0.0
            for m in range(1, horizon_months + 1):
                t_future = min(len(s_vals) - 1, t0_clean + m)
                s_future = s_vals[t_future]
                p_active = float(np.clip(s_future / s_t0, 0.0, 1.0))
                
                rev_mult += p_active
                clv_mult += p_active * margin * (monthly_df ** m)

            table[t0] = (rev_mult, clv_mult)

        self._tenure_cache[key] = table
        return table

    def calculate_customer_clv(
        self,
        monthly_charges: float,
        current_tenure: int,
        churn_probability: float,
        horizon_months: Optional[int] = None,
        margin: Optional[float] = None,
        discount_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes granular CLV components for a single customer record.
        """
        H = horizon_months or self.default_horizon
        cm = margin if margin is not None else self.default_margin
        r = discount_rate if discount_rate is not None else self.default_discount_rate

        table = self._get_multipliers_table(H, cm, r)
        t0 = max(0, min(99, int(current_tenure)))
        rev_mult, clv_mult = table.get(t0, (0.0, 0.0))

        expected_future_rev = monthly_charges * rev_mult
        discounted_clv = monthly_charges * clv_mult
        expected_future_margin = expected_future_rev * cm
        revenue_at_risk = discounted_clv * churn_probability

        return {
            "forecast_horizon_months": H,
            "contribution_margin_pct": round(cm, 4),
            "annual_discount_rate": round(r, 4),
            "expected_future_revenue": round(expected_future_rev, 2),
            "expected_future_margin": round(expected_future_margin, 2),
            "discounted_clv": round(discounted_clv, 2),
            "revenue_at_risk": round(revenue_at_risk, 2),
        }

    def estimate_dataset_clv(
        self,
        df: pd.DataFrame,
        churn_probabilities: Optional[np.ndarray] = None,
        horizon_months: Optional[int] = None,
        margin: Optional[float] = None,
        discount_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Vectorized CLV forecast across an entire customer dataset.
        Sub-second performance across thousands of customers.
        """
        out = df.copy()
        H = horizon_months or self.default_horizon
        cm = margin if margin is not None else self.default_margin
        r = discount_rate if discount_rate is not None else self.default_discount_rate

        if churn_probabilities is not None:
            probs = np.array(churn_probabilities)
        elif "churn_probability" in out.columns:
            probs = out["churn_probability"].values
        elif "churn" in out.columns:
            probs = out["churn"].values
        else:
            probs = np.full(len(out), 0.25)

        table = self._get_multipliers_table(H, cm, r)
        
        tenures = np.clip(out.get("tenure", 0).values.astype(int), 0, 99)
        monthly_charges = out.get("monthly_charges", 50.0).values.astype(float)

        rev_multipliers = np.array([table[t][0] for t in tenures])
        clv_multipliers = np.array([table[t][1] for t in tenures])

        expected_future_rev = monthly_charges * rev_multipliers
        discounted_clv = monthly_charges * clv_multipliers
        expected_future_margin = expected_future_rev * cm
        revenue_at_risk = discounted_clv * probs

        customer_ids = out["customer_id"].values if "customer_id" in out.columns else [f"CUST-{i:05d}" for i in range(len(out))]

        return pd.DataFrame({
            "customer_id": customer_ids,
            "forecast_horizon_months": H,
            "contribution_margin_pct": round(cm, 4),
            "annual_discount_rate": round(r, 4),
            "expected_future_revenue": np.round(expected_future_rev, 2),
            "expected_future_margin": np.round(expected_future_margin, 2),
            "discounted_clv": np.round(discounted_clv, 2),
            "revenue_at_risk": np.round(revenue_at_risk, 2),
        })
