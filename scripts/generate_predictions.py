"""
Batch Prediction & Persistence Pipeline.
Generates Churn Probabilities, Survival Horizons, CLV Valuations, and Cluster Segments
for all customers and persists results into the PostgreSQL database.
Usage:
    python scripts/generate_predictions.py
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
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.connection import get_engine
from src.database.schema import Customer, CustomerSnapshot
from src.database.repository import DatabaseRepository
from src.models.churn import ChurnModelService
from src.models.survival import SurvivalAnalysisService
from src.models.clv import CLVEstimator
from src.models.segmentation import CustomerSegmentationService
from src.business.roi import simulate_retention_campaign
from src.data.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_all_predictions():
    logger.info("Initializing Batch Prediction Workflow...")
    
    # 1. Load trained models
    churn_service = ChurnModelService.load()
    survival_service = SurvivalAnalysisService.load()
    clv_estimator = CLVEstimator(survival_service)
    segment_service = CustomerSegmentationService.load()
    repo = DatabaseRepository()

    # 2. Extract customer records from database
    engine = get_engine()
    query = """
    SELECT 
        c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
        c.phone_service, c.multiple_lines, c.internet_service, c.online_security,
        c.online_backup, c.device_protection, c.tech_support, c.streaming_tv,
        c.streaming_movies, c.contract, c.paperless_billing, c.payment_method,
        s.tenure, s.monthly_charges, s.total_charges, s.churn_status as churn
    FROM customers c
    JOIN customer_snapshots s ON c.customer_id = s.customer_id
    """
    df_customers = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df_customers)} customers from database for scoring.")

    if len(df_customers) == 0:
        logger.error("No customers found in database. Run 'python scripts/import_dataset.py' first.")
        sys.exit(1)

    # 2b. Obtain valid model_run_id from database or create scoring record
    with Session(engine) as session:
        from src.database.schema import ModelRun
        latest_run = session.query(ModelRun).order_by(ModelRun.created_at.desc()).first()
        if latest_run:
            model_run_id = latest_run.model_run_id
            logger.info(f"Associating predictions with existing model run: {model_run_id}")
        else:
            model_run_id = f"SCORE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            repo.record_model_run(
                model_run_id=model_run_id,
                model_type="LightGBM+CoxPH+KMeans",
                model_version="1.0.0",
                threshold=churn_service.best_threshold,
            )
            logger.info(f"Registered new model run in database: {model_run_id}")

    # 3. Churn Predictions
    logger.info("Scoring customer churn probabilities...")
    churn_probs = churn_service.predict_proba(df_customers)
    binary_preds = (churn_probs >= churn_service.best_threshold).astype(int)

    df_churn_preds = pd.DataFrame({
        "customer_id": df_customers["customer_id"],
        "churn_probability": np.round(churn_probs, 4),
        "predicted_churn": binary_preds,
    })
    repo.save_churn_predictions(df_churn_preds, model_run_id)

    # 4. Survival Predictions
    logger.info("Scoring customer conditional survival retention horizons...")
    with Session(engine) as session:
        from src.database.schema import SurvivalPrediction
        from sqlalchemy import delete
        session.execute(delete(SurvivalPrediction).where(SurvivalPrediction.model_run_id == model_run_id))
        
        surv_records = []
        is_med_inf = np.isinf(survival_service.median_survival_) or np.isnan(survival_service.median_survival_)
        for _, row in df_customers.iterrows():
            cid = str(row["customer_id"])
            tenure = int(row["tenure"])
            cond_surv = survival_service.predict_conditional_survival(tenure, [12, 24, 36, 48])
            
            med_rem = None if is_med_inf else max(0.0, float(survival_service.median_survival_ - tenure))
            
            surv_records.append(SurvivalPrediction(
                customer_id=cid,
                model_run_id=model_run_id,
                median_remaining_tenure=med_rem,
                prob_active_12m=cond_surv.get(12, 0.0),
                prob_active_24m=cond_surv.get(24, 0.0),
                prob_active_36m=cond_surv.get(36, 0.0),
                prob_active_48m=cond_surv.get(48, 0.0),
            ))
            if len(surv_records) % 1000 == 0:
                session.add_all(surv_records)
                session.commit()
                surv_records = []
        if surv_records:
            session.add_all(surv_records)
            session.commit()
    logger.info("Saved survival predictions.")

    # 5. Customer Lifetime Value (CLV)
    logger.info("Scoring Customer Lifetime Values (CLV)...")
    df_clv = clv_estimator.estimate_dataset_clv(df_customers, churn_probabilities=churn_probs)
    repo.save_clv_estimates(df_clv, model_run_id)

    # 6. Customer Segmentation
    logger.info("Assigning customer behavioral segments...")
    df_seg_input = df_customers.copy()
    df_seg_input["churn_probability"] = churn_probs
    df_seg_input["discounted_clv"] = df_clv["discounted_clv"].values
    df_seg_input = engineer_features(df_seg_input)

    df_segmented = segment_service.transform(df_seg_input)
    repo.save_segments(df_segmented, model_run_id)

    # 7. Benchmark Retention Campaigns
    logger.info("Simulating baseline retention campaigns...")
    avg_mc = float(df_customers["monthly_charges"].mean())
    high_risk_count = int(np.sum(binary_preds == 1))

    # Campaign 1: Targeted ML Retention
    camp_ml = simulate_retention_campaign(
        eligible_count=high_risk_count,
        avg_monthly_charges=avg_mc * 1.15,
        discount_pct=0.20,
        discount_duration_months=3,
        offer_acceptance_rate=0.45,
        baseline_churn_rate=0.65,
        incremental_uplift_rate=0.28,
    )
    camp_ml_data = {
        "campaign_id": "CAMP-ML-TARGETED",
        "campaign_name": "Targeted ML Retention Campaign",
        "target_risk_tier": "High Risk (ML Tuned)",
        "discount_pct": 0.20,
        "discount_duration_months": 3,
        "offer_acceptance_rate": 0.45,
        "expected_uplift_pct": 0.28,
        "budget_limit": 50000.0,
        "eligible_customers": camp_ml["eligible_customers"],
        "estimated_retained_customers": camp_ml["incremental_retained_customers"],
        "campaign_cost": camp_ml["total_campaign_cost"],
        "retained_contribution": camp_ml["total_retained_contribution"],
        "net_benefit": camp_ml["net_benefit"],
        "roi_pct": camp_ml["roi_pct"],
    }
    repo.save_retention_campaign(camp_ml_data)

    # Campaign 2: Broad Mass Discount
    camp_broad = simulate_retention_campaign(
        eligible_count=len(df_customers),
        avg_monthly_charges=avg_mc,
        discount_pct=0.15,
        discount_duration_months=3,
        offer_acceptance_rate=0.50,
        baseline_churn_rate=0.265,
        incremental_uplift_rate=0.12,
    )
    camp_broad_data = {
        "campaign_id": "CAMP-BROAD-MASS",
        "campaign_name": "Broad Mass Discount (Untargeted)",
        "target_risk_tier": "All Customers",
        "discount_pct": 0.15,
        "discount_duration_months": 3,
        "offer_acceptance_rate": 0.50,
        "expected_uplift_pct": 0.12,
        "budget_limit": 100000.0,
        "eligible_customers": camp_broad["eligible_customers"],
        "estimated_retained_customers": camp_broad["incremental_retained_customers"],
        "campaign_cost": camp_broad["total_campaign_cost"],
        "retained_contribution": camp_broad["total_retained_contribution"],
        "net_benefit": camp_broad["net_benefit"],
        "roi_pct": camp_broad["roi_pct"],
    }
    repo.save_retention_campaign(camp_broad_data)

    # 8. Query and display summary
    summary = repo.get_dashboard_summary()
    print("\n" + "=" * 65)
    print("PREDICTION & DATABASE PERSISTENCE COMPLETE")
    print("=" * 65)
    print(f"Total Customers Scored        : {summary['total_customers']}")
    print(f"High-Risk Predicted Customers : {summary['high_risk_customers_predicted']} (Threshold: {churn_service.best_threshold:.3f})")
    print(f"Average Estimated CLV         : ${summary['average_estimated_clv']:,.2f}")
    print(f"Total Revenue At Risk         : ${summary['total_revenue_at_risk']:,.2f}")
    print(f"Targeted Campaign Net Benefit : ${camp_ml['net_benefit']:,.2f} (ROI: {camp_ml['roi_pct']}%)")
    print(f"Broad Campaign Net Benefit    : ${camp_broad['net_benefit']:,.2f} (ROI: {camp_broad['roi_pct']}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_all_predictions()
