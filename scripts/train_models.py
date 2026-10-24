"""
End-to-End Model Training Pipeline.
Trains Churn Classifier (LightGBM + Baseline), fits Lifelines Survival models,
caches SHAP global feature importances, fits K-Means segmentation, and logs model run to database.
Usage:
    python scripts/train_models.py
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
import pandas as pd
import numpy as np

from src.data.load_data import load_raw_data, standardize_columns
from src.data.validate_data import validate_raw_dataframe
from src.data.preprocess import split_data
from src.models.churn import ChurnModelService
from src.models.shap_explainer import ShapExplanationService
from src.models.survival import SurvivalAnalysisService
from src.models.clv import CLVEstimator
from src.models.segmentation import CustomerSegmentationService
from src.database.repository import DatabaseRepository
from src.config import MODEL_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_all_models():
    logger.info("=== STEP 1: LOADING & VALIDATING DATASET ===")
    df_raw = load_raw_data()
    val_report = validate_raw_dataframe(df_raw)
    if not val_report["is_valid"]:
        raise ValueError(f"Dataset invalid: {val_report['issues']}")
    
    df_clean = standardize_columns(df_raw)
    logger.info(f"Loaded {len(df_clean)} cleaned customer records.")

    logger.info("=== STEP 2: STRATIFIED TRAIN / VAL / TEST SPLIT ===")
    train_df, val_df, test_df = split_data(df_clean)

    logger.info("=== STEP 3: TRAINING CHURN CLASSIFIER (LIGHTGBM & BASELINE) ===")
    churn_service = ChurnModelService()
    churn_metrics = churn_service.train(train_df, val_df, test_df)
    churn_service.save()

    logger.info("=== STEP 4: COMPUTING & CACHING GLOBAL SHAP VALUES ===")
    shap_service = ShapExplanationService(churn_service)
    # Use a representative sample of test data for SHAP global caching
    shap_sample = test_df.sample(min(500, len(test_df)), random_state=42)
    shap_service.compute_and_cache_global_shap(shap_sample)

    logger.info("=== STEP 5: FITTING LIFELINES SURVIVAL MODELS ===")
    survival_service = SurvivalAnalysisService()
    survival_service.fit(train_df)
    survival_service.save()

    logger.info("=== STEP 6: FITTING CUSTOMER SEGMENTATION ===")
    # Prepare features for segmentation
    clv_est = CLVEstimator(survival_service)
    train_probs = churn_service.predict_proba(train_df)
    train_clv = clv_est.estimate_dataset_clv(train_df, churn_probabilities=train_probs)
    
    train_seg_input = train_df.copy()
    train_seg_input["churn_probability"] = train_probs
    train_seg_input["discounted_clv"] = train_clv["discounted_clv"].values
    
    from src.data.features import engineer_features
    train_seg_input = engineer_features(train_seg_input)

    seg_service = CustomerSegmentationService()
    seg_service.fit(train_seg_input)
    seg_service.save()

    logger.info("=== STEP 7: LOGGING MODEL RUN TO DATABASE ===")
    model_run_id = f"RUN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    repo = DatabaseRepository()
    repo.record_model_run(
        model_run_id=model_run_id,
        model_type="LightGBM+CoxPH+KMeans",
        model_version="1.0.0",
        threshold=churn_service.best_threshold,
        metrics=churn_metrics,
        hyperparameters={
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "scale_pos_weight": churn_metrics.get("scale_pos_weight", 1.5),
            "survival_c_index": float(survival_service.cph.concordance_index_),
            "segmentation_k": seg_service.n_clusters,
            "segmentation_silhouette": seg_service.silhouette_scores.get(seg_service.n_clusters, 0.0),
        },
    )

    print("\n" + "=" * 65)
    print("MODEL TRAINING & EVALUATION REPORT")
    print("=" * 65)
    print(f"Model Run ID         : {model_run_id}")
    print(f"Selected Threshold   : {churn_service.best_threshold:.3f} (Validation Optimized)")
    
    tuned = churn_metrics["lightgbm_tuned_threshold"]
    base = churn_metrics["baseline_logistic_regression"]
    std = churn_metrics["lightgbm_standard_threshold_0_50"]

    print("\nHELD-OUT TEST EVALUATION COMPARISON:")
    print("-" * 65)
    print(f"{'Metric':<22} | {'LightGBM (Tuned)':<18} | {'LightGBM (0.50)':<16} | {'Logistic Reg Baseline'}")
    print("-" * 65)
    for m in ["roc_auc", "pr_auc", "f1_score", "recall", "precision", "accuracy", "brier_score"]:
        print(f"{m:<22} | {tuned.get(m, 0.0):<18.4f} | {std.get(m, 0.0):<16.4f} | {base.get(m, 0.0):.4f}")
    
    print("-" * 65)
    print(f"Survival Concordance : {survival_service.cph.concordance_index_:.4f}")
    print(f"Median Survival Time : {survival_service.median_survival_:.1f} months")
    print(f"Optimal Clusters (K) : {seg_service.n_clusters} (Silhouette: {seg_service.silhouette_scores.get(seg_service.n_clusters, 0.0):.4f})")
    print("=" * 65 + "\n")

    return model_run_id


if __name__ == "__main__":
    train_all_models()
