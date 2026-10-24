"""
Survival Analysis Module using Lifelines.
Distinguishes observed tenure, churn event, and right-censored active customers.
Fits Kaplan-Meier survival curves and Cox Proportional Hazards models with conditional survival probabilities.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import joblib
import logging

from src.config import MODEL_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SURVIVAL_LIMITATION_DISCLAIMER = (
    "Methodological Caveat: The IBM Telco dataset provides observed tenure at a single point in time "
    "and whether the customer has churned. It is right-censored for active customers (event=0). "
    "Survival curves provide estimated statistical retention probabilities over time, NOT exact calendar departure dates. "
    "Cox proportional hazards coefficients indicate relative statistical hazard ratios, not causal proofs."
)


class SurvivalAnalysisService:
    """Estimates customer duration dynamics, median survival, and conditional horizon retention."""

    def __init__(self):
        self.kmf: Optional[KaplanMeierFitter] = None
        self.cph: Optional[CoxPHFitter] = None
        self.group_kmfs: Dict[str, Dict[str, KaplanMeierFitter]] = {}
        self.median_survival_: float = np.nan
        self.cph_columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> "SurvivalAnalysisService":
        """
        Fits Kaplan-Meier and Cox Proportional Hazards models on tenure and churn.
        Right-censoring is handled explicitly: churn=1 is event, churn=0 is censored.
        """
        logger.info("Fitting Kaplan-Meier Overall Survival Model...")
        # Clean tenure to strictly positive for survival models
        df_surv = df.copy()
        df_surv["duration"] = np.maximum(df_surv["tenure"].values, 1)
        df_surv["event"] = df_surv["churn"].astype(int).values

        # 1. Global Kaplan-Meier
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(
            durations=df_surv["duration"],
            event_observed=df_surv["event"],
            label="Overall Telco Customer Cohort",
        )
        self.median_survival_ = float(self.kmf.median_survival_time_)
        logger.info(f"Fitted Kaplan-Meier. Median survival time: {self.median_survival_} months.")

        # 2. Stratified Kaplan-Meier by Contract Type
        if "contract" in df_surv.columns:
            self.group_kmfs["contract"] = {}
            for contract_type in df_surv["contract"].unique():
                mask = df_surv["contract"] == contract_type
                kmf_grp = KaplanMeierFitter()
                kmf_grp.fit(
                    durations=df_surv.loc[mask, "duration"],
                    event_observed=df_surv.loc[mask, "event"],
                    label=f"Contract: {contract_type}",
                )
                self.group_kmfs["contract"][contract_type] = kmf_grp

        # 3. Stratified by Internet Service
        if "internet_service" in df_surv.columns:
            self.group_kmfs["internet_service"] = {}
            for service in df_surv["internet_service"].unique():
                mask = df_surv["internet_service"] == service
                kmf_grp = KaplanMeierFitter()
                kmf_grp.fit(
                    durations=df_surv.loc[mask, "duration"],
                    event_observed=df_surv.loc[mask, "event"],
                    label=f"Internet: {service}",
                )
                self.group_kmfs["internet_service"][service] = kmf_grp

        # 4. Cox Proportional Hazards Model
        logger.info("Fitting Cox Proportional Hazards Model...")
        cph_cols = ["duration", "event", "monthly_charges"]
        
        # Add key categorical dummy variables
        cph_df = df_surv[["duration", "event", "monthly_charges"]].copy()
        
        if "contract" in df_surv.columns:
            contract_dummies = pd.get_dummies(df_surv["contract"], prefix="contract", drop_first=True)
            cph_df = pd.concat([cph_df, contract_dummies], axis=1)

        if "internet_service" in df_surv.columns:
            internet_dummies = pd.get_dummies(df_surv["internet_service"], prefix="internet", drop_first=True)
            cph_df = pd.concat([cph_df, internet_dummies], axis=1)

        if "paperless_billing" in df_surv.columns:
            cph_df["paperless"] = (df_surv["paperless_billing"].astype(str).str.lower() == "yes").astype(int)

        self.cph = CoxPHFitter(penalizer=0.01)
        self.cph.fit(cph_df, duration_col="duration", event_col="event")
        self.cph_columns = [c for c in cph_df.columns if c not in ["duration", "event"]]
        logger.info(f"Fitted Cox Proportional Hazards Model. Concordance Index: {self.cph.concordance_index_:.4f}")

        return self

    def get_survival_table(self) -> pd.DataFrame:
        """Returns overall Kaplan-Meier survival probabilities across key timeline intervals."""
        if self.kmf is None:
            raise ValueError("Model is not fitted.")
        
        timeline = [1, 6, 12, 24, 36, 48, 60, 72]
        probs = self.kmf.survival_function_at_times(timeline).values
        ci = self.kmf.confidence_interval_survival_function_.loc[timeline]

        return pd.DataFrame({
            "tenure_months": timeline,
            "survival_probability": np.round(probs, 4),
            "ci_lower": np.round(ci.iloc[:, 0].values, 4),
            "ci_upper": np.round(ci.iloc[:, 1].values, 4),
        })

    def predict_conditional_survival(
        self,
        current_tenure: int,
        future_horizons: List[int] = [12, 24, 36, 48],
    ) -> Dict[int, float]:
        """
        Calculates conditional survival probability:
        P(T > current_tenure + delta_t | T > current_tenure) = S(current_tenure + delta_t) / S(current_tenure)
        """
        if self.kmf is None:
            raise ValueError("Kaplan-Meier model is not fitted.")

        t0 = max(1, int(current_tenure))
        s_t0 = float(self.kmf.survival_function_at_times(t0).values[0])
        
        # Guard against zero denominator
        if s_t0 <= 1e-6:
            return {h: 0.0 for h in future_horizons}

        conditional_probs = {}
        for h in future_horizons:
            t_future = min(72, t0 + h)
            s_future = float(self.kmf.survival_function_at_times(t_future).values[0])
            cond_p = s_future / s_t0
            conditional_probs[h] = round(float(np.clip(cond_p, 0.0, 1.0)), 4)

        return conditional_probs

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persists survival models to disk."""
        save_path = filepath or (MODEL_ARTIFACTS_DIR / "survival_models.joblib")
        joblib.dump(self, save_path)
        logger.info(f"Saved survival analysis models to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "SurvivalAnalysisService":
        """Loads fitted survival analysis models."""
        load_path = filepath or (MODEL_ARTIFACTS_DIR / "survival_models.joblib")
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Survival model artifact not found at {load_path}")
        return joblib.load(load_path)
