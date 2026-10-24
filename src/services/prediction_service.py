"""
Inference Service Layer.
Decouples model execution and prediction logic from the Streamlit UI and REST API.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

from src.models.churn import ChurnModelService
from src.models.shap_explainer import ShapExplanationService
from src.models.survival import SurvivalAnalysisService
from src.models.clv import CLVEstimator
from src.business.recommendations import generate_prescriptive_recommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service layer executing single and batch inference."""

    def __init__(self):
        self.churn_service = ChurnModelService.load()
        self.survival_service = SurvivalAnalysisService.load()
        self.clv_estimator = CLVEstimator(self.survival_service)
        self.shap_service = ShapExplanationService(self.churn_service)

    def predict_single_customer(self, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end inference for a single customer payload:
        1. Predicts churn probability & risk tier.
        2. Computes top positive and negative SHAP drivers.
        3. Estimates conditional survival retention horizons.
        4. Calculates forward-looking CLV and revenue at risk.
        5. Generates business prescriptive recommendation.
        """
        df_single = pd.DataFrame([customer_dict])
        
        # 1. Churn Probability
        prob = float(self.churn_service.predict_proba(df_single)[0])
        pred_churn = int(prob >= self.churn_service.best_threshold)

        if prob < 0.25:
            tier = "Low"
        elif prob < 0.50:
            tier = "Medium"
        elif prob < 0.75:
            tier = "High"
        else:
            tier = "Critical"

        # 2. Local SHAP Drivers
        shap_explanation = self.shap_service.explain_single_customer(df_single, top_k=4)

        # 3. Survival Outlook
        tenure = int(customer_dict.get("tenure", 0))
        cond_surv = self.survival_service.predict_conditional_survival(tenure, [12, 24, 36, 48])

        # 4. Customer Lifetime Value
        mc = float(customer_dict.get("monthly_charges", 50.0))
        clv_res = self.clv_estimator.calculate_customer_clv(
            monthly_charges=mc,
            current_tenure=tenure,
            churn_probability=prob,
        )

        # 5. Prescriptive Recommendation
        profile_data = {
            "churn_probability": prob,
            "discounted_clv": clv_res["discounted_clv"],
            "contract": customer_dict.get("contract", ""),
            "internet_service": customer_dict.get("internet_service", ""),
            "tech_support": customer_dict.get("tech_support", ""),
            "tenure": tenure,
        }
        recom = generate_prescriptive_recommendation(profile_data)

        return {
            "customer_id": customer_dict.get("customer_id", "PROSPECT-001"),
            "churn_probability": round(prob, 4),
            "predicted_churn": pred_churn,
            "decision_threshold": self.churn_service.best_threshold,
            "risk_tier": tier,
            "risk_label": "Elevated Risk" if pred_churn == 1 else "Normal / Retained",
            "clv": clv_res,
            "survival_horizons": cond_surv,
            "shap_explanation": {
                "risk_increasing_factors": shap_explanation["risk_increasing_factors"],
                "risk_decreasing_factors": shap_explanation["risk_decreasing_factors"],
                "disclaimer": shap_explanation["disclaimer"],
            },
            "recommendation": recom,
        }

    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes batch inference across multiple customer records."""
        df_batch = pd.DataFrame(customers)
        probs = self.churn_service.predict_proba(df_batch)
        clv_df = self.clv_estimator.estimate_dataset_clv(df_batch, churn_probabilities=probs)

        results = []
        for i, row in df_batch.iterrows():
            prob = float(probs[i])
            pred_churn = int(prob >= self.churn_service.best_threshold)
            tier = "Low" if prob < 0.25 else ("Medium" if prob < 0.50 else ("High" if prob < 0.75 else "Critical"))
            
            results.append({
                "customer_id": row.get("customer_id", f"CUST-{i:05d}"),
                "churn_probability": round(prob, 4),
                "predicted_churn": pred_churn,
                "risk_tier": tier,
                "discounted_clv": float(clv_df.iloc[i]["discounted_clv"]),
                "revenue_at_risk": float(clv_df.iloc[i]["revenue_at_risk"]),
            })
        return results
