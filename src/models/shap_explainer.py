"""
SHAP Explainability Module for Tree Models.
Computes TreeExplainer SHAP values, global feature impact summaries,
and granular individual customer churn risk drivers with explicit causal disclaimers.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import shap
import joblib
import logging

from src.config import MODEL_ARTIFACTS_DIR
from src.models.churn import ChurnModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAUSAL_DISCLAIMER = (
    "Methodological Note: SHAP values represent model feature attribution and marginal contribution "
    "within the LightGBM predictive framework. They DO NOT represent proven counterfactual causal effects. "
    "A high positive SHAP value indicates that a feature correlates strongly with elevated churn risk in the "
    "observational data, but changing that feature may not causally reduce churn."
)


class ShapExplanationService:
    """Provides global and local SHAP explainability for LightGBM churn models."""

    def __init__(self, churn_service: Optional[ChurnModelService] = None):
        self.churn_service = churn_service or ChurnModelService.load()
        self.explainer: Optional[shap.TreeExplainer] = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes TreeExplainer with the trained LightGBM model."""
        if self.churn_service.lgb_model is None:
            raise ValueError("Churn model is not trained or loaded.")
        self.explainer = shap.TreeExplainer(self.churn_service.lgb_model)
        logger.info("Initialized TreeExplainer for LightGBM model.")

    def compute_and_cache_global_shap(
        self,
        sample_df: pd.DataFrame,
        cache_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Computes SHAP values across a representative sample and caches them for fast dashboard loading.
        """
        target_path = cache_path or (MODEL_ARTIFACTS_DIR / "shap_global_cache.joblib")
        
        # Transform sample features
        X_sample, feature_names = self.churn_service.preprocessor.transform(sample_df)
        shap_values = self.explainer.shap_values(X_sample)

        # In binary classification, shap_values is an array of shape (N, features) or list of two arrays
        if isinstance(shap_values, list):
            # Class 1 (churn) SHAP values
            shap_matrix = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        # Mean absolute SHAP importance
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        cached_data = {
            "importance_df": importance_df,
            "shap_matrix": shap_matrix,
            "sample_features": X_sample,
            "feature_names": feature_names,
            "base_value": float(np.mean(self.explainer.expected_value)) if hasattr(self.explainer, "expected_value") else 0.0,
            "disclaimer": CAUSAL_DISCLAIMER,
        }

        joblib.dump(cached_data, target_path)
        logger.info(f"Cached global SHAP values to {target_path}")
        return cached_data

    def get_global_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Retrieves top global predictive features sorted by impact."""
        cache_path = MODEL_ARTIFACTS_DIR / "shap_global_cache.joblib"
        if cache_path.exists():
            cached = joblib.load(cache_path)
            return cached["importance_df"].head(top_n)
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    def explain_single_customer(
        self,
        customer_df: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Computes granular customer-level risk drivers.
        Returns top positive factors (raising churn risk) and negative factors (lowering churn risk).
        """
        if len(customer_df) != 1:
            raise ValueError("customer_df must contain exactly 1 row for single customer explanation.")

        X, feature_names = self.churn_service.preprocessor.transform(customer_df)
        pred_prob = float(self.churn_service.lgb_model.predict_proba(X)[0, 1])

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        # Pair feature names, values, and SHAP scores
        drivers = []
        for feat_name, feat_val, s_val in zip(feature_names, X[0], sv):
            drivers.append({
                "feature": feat_name,
                "feature_value": round(float(feat_val), 3),
                "shap_impact": round(float(s_val), 4),
            })

        df_drivers = pd.DataFrame(drivers)
        
        # Factors pushing towards churn (positive SHAP impact)
        risk_increasing = (
            df_drivers[df_drivers["shap_impact"] > 0]
            .sort_values(by="shap_impact", ascending=False)
            .head(top_k)
            .to_dict(orient="records")
        )

        # Factors pushing away from churn (negative SHAP impact)
        risk_decreasing = (
            df_drivers[df_drivers["shap_impact"] < 0]
            .sort_values(by="shap_impact", ascending=True)
            .head(top_k)
            .to_dict(orient="records")
        )

        return {
            "predicted_probability": round(pred_prob, 4),
            "base_value": float(np.mean(self.explainer.expected_value)) if hasattr(self.explainer, "expected_value") else 0.0,
            "risk_increasing_factors": risk_increasing,
            "risk_decreasing_factors": risk_decreasing,
            "all_driver_scores": drivers,
            "disclaimer": CAUSAL_DISCLAIMER,
        }
