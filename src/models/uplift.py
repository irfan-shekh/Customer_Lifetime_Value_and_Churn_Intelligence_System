"""
Uplift Modeling and Heterogeneous Treatment Effect (HTE) Module.
Identifies 'Persuadables' vs 'Sure Things' vs 'Lost Causes'.
Provides S-Learner, T-Learner, and X-Learner meta-learners with explicit synthetic RCT demonstration workflows
and transparent methodological disclaimers regarding observational limitations.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib
import logging

from src.config import RANDOM_SEED, MODEL_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLIFT_METHODOLOGY_DISCLAIMER = (
    "CRITICAL CAUSAL INFERENCE NOTICE: The standard IBM Telco Customer Churn dataset is strictly OBSERVATIONAL. "
    "It contains NO randomized retention campaign interventions, NO treatment exposure flags, and NO counterfactual outcomes. "
    "Predictive churn models (LightGBM) identify customers likely to leave, NOT customers who will change their decision "
    "when receiving an incentive. To demonstrate uplift modeling, this system provides an explicitly simulated "
    "Randomized Controlled Trial (RCT) module based on Meta-Learners (T-Learner and S-Learner). "
    "Results from this simulated dataset MUST NOT be interpreted as real-world empirical causal validation. "
    "Real treatment effects require launching an A/B test with randomized holdout groups."
)


def generate_demonstration_rct_dataset(
    df: pd.DataFrame,
    treatment_ratio: float = 0.50,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates an EXPLICITLY LABELED synthetic Randomized Controlled Trial (RCT) dataset.
    Simulates a retention offer (e.g., 20% discount for 3 months) with heterogeneous treatment effects.
    """
    np.random.seed(random_state)
    rct_df = df.copy()
    n = len(rct_df)

    # 1. Completely randomized treatment assignment (A/B testing standard)
    treatment = (np.random.rand(n) < treatment_ratio).astype(int)
    rct_df["treatment_offer"] = treatment

    # 2. Baseline churn risk estimation
    is_m2m = (rct_df.get("contract", "").astype(str).str.lower() == "month-to-month").astype(int)
    has_fiber = (rct_df.get("internet_service", "").astype(str).str.lower() == "fiber optic").astype(int)
    tenure_norm = np.clip(rct_df.get("tenure", 12).values / 72.0, 0.0, 1.0)
    monthly_norm = np.clip(rct_df.get("monthly_charges", 50.0).values / 120.0, 0.0, 1.0)

    # Baseline probability of churn without treatment
    logit_base = -1.2 + 1.8 * is_m2m + 0.8 * has_fiber + 0.6 * monthly_norm - 2.2 * tenure_norm
    p_churn_control = 1.0 / (1.0 + np.exp(-logit_base))

    # 3. Heterogeneous Treatment Effect (Uplift = reduction in churn probability when treated)
    # Persuadable profile: Month-to-month contract, higher monthly spend, moderate tenure
    tau_uplift = (0.15 * is_m2m + 0.10 * has_fiber + 0.05 * monthly_norm) * (1.0 - 0.4 * tenure_norm)
    tau_uplift = np.clip(tau_uplift, -0.05, 0.35)  # Can have small negative uplift for 'sleeping dogs'

    # Probability of churn under treatment: P(Churn|T=1) = P(Churn|T=0) - Uplift
    p_churn_treated = np.clip(p_churn_control - tau_uplift, 0.01, 0.99)

    # 4. Realized observed outcome given assigned treatment
    actual_p_churn = np.where(treatment == 1, p_churn_treated, p_churn_control)
    realized_churn = (np.random.rand(n) < actual_p_churn).astype(int)

    rct_df["synthetic_rct_churn_outcome"] = realized_churn
    rct_df["true_synthetic_uplift"] = np.round(tau_uplift, 4)
    rct_df["is_synthetic_demonstration"] = True

    logger.info(
        f"Generated Demonstration RCT: Control Churn={p_churn_control.mean():.1%}, "
        f"Treated Churn={p_churn_treated.mean():.1%}, "
        f"Average True Uplift={tau_uplift.mean():.1%}"
    )

    return rct_df


class TLearnerUpliftModel:
    """
    Two-Model (T-Learner) Uplift Estimator.
    Trains model_0 on control group (W=0) and model_1 on treated group (W=1).
    Individual Treatment Effect (ITE / Uplift) = P(Churn|W=0) - P(Churn|W=1).
    Positive uplift indicates churn reduction benefit from retention offer.
    """

    def __init__(self):
        self.model_control = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        self.model_treated = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> "TLearnerUpliftModel":
        """Fits separate estimators for control and treated sub-cohorts."""
        self.feature_cols = X.columns.tolist()

        ctrl_mask = (treatment == 0)
        treat_mask = (treatment == 1)

        if ctrl_mask.sum() == 0 or treat_mask.sum() == 0:
            raise ValueError("Both treatment and control cohorts must have samples.")

        # Fit control estimator
        self.model_control.fit(X[ctrl_mask], outcome[ctrl_mask])

        # Fit treated estimator
        self.model_treated.fit(X[treat_mask], outcome[treat_mask])

        self.is_fitted = True
        logger.info(
            f"TLearner fitted successfully: Control N={ctrl_mask.sum()}, Treated N={treat_mask.sum()}"
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimates individual treatment effect (uplift score).
        Positive score = expected reduction in churn probability if offered retention incentive.
        """
        if not self.is_fitted:
            raise ValueError("TLearner must be fitted before predict_uplift.")
        
        prob_churn_ctrl = self.model_control.predict_proba(X[self.feature_cols])[:, 1]
        prob_churn_treat = self.model_treated.predict_proba(X[self.feature_cols])[:, 1]
        
        # Uplift = P(Churn|Control) - P(Churn|Treated)
        uplift = prob_churn_ctrl - prob_churn_treat
        return np.round(uplift, 4)

    def categorize_customers(
        self,
        X: pd.DataFrame,
        baseline_churn_prob: np.ndarray,
        uplift_scores: np.ndarray,
    ) -> List[str]:
        """
        Segments customers into classical uplift quadrants:
        - Persuadables: High baseline churn risk AND positive uplift from offer.
        - Sure Things: Low baseline churn risk (will stay without spending campaign budget).
        - Lost Causes: High baseline churn risk but negligible uplift (offer fails to retain).
        - Sleeping Dogs: Negative uplift (intervention triggers contract cancellation).
        """
        categories = []
        median_churn = float(np.median(baseline_churn_prob))
        
        for p_base, up in zip(baseline_churn_prob, uplift_scores):
            if up < -0.02:
                categories.append("Sleeping Dogs (Do Not Disturb)")
            elif up >= 0.08 and p_base >= median_churn:
                categories.append("Persuadables (Target)")
            elif p_base < median_churn and up < 0.08:
                categories.append("Sure Things (Organic Stayers)")
            else:
                categories.append("Lost Causes (Low Response)")

        return categories
