"""
Database repository / Data Access Object (DAO) module.
Provides type-safe, parameterized CRUD operations for customers, snapshots,
model predictions, CLV valuations, and retention campaigns.
Avoids duplicate records upon repeated ingestion.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import select, func, and_, or_, delete, text
from sqlalchemy.orm import Session
import logging

from src.database.connection import get_engine, initialize_schema
from src.database.schema import (
    Customer,
    CustomerSnapshot,
    ChurnPrediction,
    SurvivalPrediction,
    CustomerLifetimeValue,
    CustomerSegment,
    RetentionCampaign,
    CampaignAssignment,
    ModelRun,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository handling all database persistence operations."""

    def __init__(self):
        self.engine = get_engine()
        self.is_connected = (self.engine is not None)
        if self.is_connected:
            try:
                initialize_schema()
            except Exception as e:
                logger.warning(f"Could not initialize schema: {e}")

    def record_model_run(
        self,
        model_run_id: str,
        model_type: str,
        model_version: str,
        threshold: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persists model run execution metadata."""
        with Session(self.engine) as session:
            existing = session.get(ModelRun, model_run_id)
            if existing:
                existing.threshold = threshold
                existing.metrics = metrics
                existing.hyperparameters = hyperparameters
            else:
                run = ModelRun(
                    model_run_id=model_run_id,
                    model_type=model_type,
                    model_version=model_version,
                    threshold=threshold,
                    metrics=metrics,
                    hyperparameters=hyperparameters,
                )
                session.add(run)
            session.commit()
            return model_run_id

    def bulk_upsert_customers(self, df: pd.DataFrame) -> int:
        """
        Upserts customers and customer snapshots.
        Idempotent: Avoids duplicate records when importing the same dataset repeatedly.
        """
        records_loaded = 0
        with Session(self.engine) as session:
            for _, row in df.iterrows():
                cid = str(row["customer_id"])
                cust = session.get(Customer, cid)
                
                if not cust:
                    cust = Customer(customer_id=cid)
                    session.add(cust)
                
                # Update attributes
                cust.gender = str(row.get("gender", ""))
                cust.senior_citizen = int(row.get("senior_citizen", 0))
                cust.partner = str(row.get("partner", ""))
                cust.dependents = str(row.get("dependents", ""))
                cust.phone_service = str(row.get("phone_service", ""))
                cust.multiple_lines = str(row.get("multiple_lines", ""))
                cust.internet_service = str(row.get("internet_service", ""))
                cust.online_security = str(row.get("online_security", ""))
                cust.online_backup = str(row.get("online_backup", ""))
                cust.device_protection = str(row.get("device_protection", ""))
                cust.tech_support = str(row.get("tech_support", ""))
                cust.streaming_tv = str(row.get("streaming_tv", ""))
                cust.streaming_movies = str(row.get("streaming_movies", ""))
                cust.contract = str(row.get("contract", ""))
                cust.paperless_billing = str(row.get("paperless_billing", ""))
                cust.payment_method = str(row.get("payment_method", ""))

                # Snapshot logic
                snap = session.execute(
                    select(CustomerSnapshot).where(CustomerSnapshot.customer_id == cid)
                ).scalars().first()

                if not snap:
                    snap = CustomerSnapshot(
                        customer_id=cid,
                        tenure=int(row.get("tenure", 0)),
                        monthly_charges=float(row.get("monthly_charges", 0.0)),
                        total_charges=float(row.get("total_charges", 0.0)),
                        churn_status=int(row.get("churn", 0)),
                        churn_reason=str(row.get("churn_reason", "")) if pd.notna(row.get("churn_reason")) else None,
                    )
                    session.add(snap)
                else:
                    snap.tenure = int(row.get("tenure", 0))
                    snap.monthly_charges = float(row.get("monthly_charges", 0.0))
                    snap.total_charges = float(row.get("total_charges", 0.0))
                    snap.churn_status = int(row.get("churn", 0))
                    snap.churn_reason = str(row.get("churn_reason", "")) if pd.notna(row.get("churn_reason")) else None

                records_loaded += 1
                if records_loaded % 1000 == 0:
                    session.commit()
            
            session.commit()
            logger.info(f"Upserted {records_loaded} customer records.")
            return records_loaded

    def save_churn_predictions(
        self,
        predictions_df: pd.DataFrame,
        model_run_id: str,
    ) -> int:
        """Saves batch churn risk estimates with risk tier categorization."""
        count = 0
        with Session(self.engine) as session:
            # Clear old predictions for this model run if re-running
            session.execute(delete(ChurnPrediction).where(ChurnPrediction.model_run_id == model_run_id))
            
            for _, row in predictions_df.iterrows():
                cid = str(row["customer_id"])
                prob = float(row["churn_probability"])
                pred = int(row["predicted_churn"])
                
                # Determine risk tier
                if prob < 0.25:
                    tier = "Low"
                elif prob < 0.50:
                    tier = "Medium"
                elif prob < 0.75:
                    tier = "High"
                else:
                    tier = "Critical"

                drivers = row.get("top_churn_drivers", None)
                pred_obj = ChurnPrediction(
                    customer_id=cid,
                    model_run_id=model_run_id,
                    churn_probability=round(prob, 4),
                    predicted_churn=pred,
                    risk_tier=tier,
                    top_churn_drivers=drivers,
                )
                session.add(pred_obj)
                count += 1
                if count % 1000 == 0:
                    session.commit()
            session.commit()
            logger.info(f"Saved {count} churn predictions.")
            return count

    def save_clv_estimates(
        self,
        clv_df: pd.DataFrame,
        model_run_id: str,
    ) -> int:
        """Saves customer lifetime value forecasts and revenue at risk."""
        count = 0
        with Session(self.engine) as session:
            session.execute(delete(CustomerLifetimeValue).where(CustomerLifetimeValue.model_run_id == model_run_id))
            
            for _, row in clv_df.iterrows():
                cid = str(row["customer_id"])
                clv_obj = CustomerLifetimeValue(
                    customer_id=cid,
                    model_run_id=model_run_id,
                    forecast_horizon_months=int(row["forecast_horizon_months"]),
                    contribution_margin_pct=float(row["contribution_margin_pct"]),
                    annual_discount_rate=float(row["annual_discount_rate"]),
                    expected_future_revenue=round(float(row["expected_future_revenue"]), 2),
                    expected_future_margin=round(float(row["expected_future_margin"]), 2),
                    discounted_clv=round(float(row["discounted_clv"]), 2),
                    revenue_at_risk=round(float(row["revenue_at_risk"]), 2),
                )
                session.add(clv_obj)
                count += 1
                if count % 1000 == 0:
                    session.commit()
            session.commit()
            logger.info(f"Saved {count} CLV estimates.")
            return count

    def save_segments(
        self,
        segments_df: pd.DataFrame,
        model_run_id: str,
    ) -> int:
        """Saves customer segmentation cluster assignments."""
        count = 0
        with Session(self.engine) as session:
            session.execute(delete(CustomerSegment).where(CustomerSegment.model_run_id == model_run_id))
            
            for _, row in segments_df.iterrows():
                cid = str(row["customer_id"])
                seg = CustomerSegment(
                    customer_id=cid,
                    model_run_id=model_run_id,
                    cluster_id=int(row["cluster_id"]),
                    cluster_label=str(row["cluster_label"]),
                    cluster_description=str(row.get("cluster_description", "")),
                )
                session.add(seg)
                count += 1
            session.commit()
            logger.info(f"Saved {count} customer segment assignments.")
            return count

    def save_retention_campaign(self, campaign_data: Dict[str, Any]) -> str:
        """Persists retention campaign simulation scenario."""
        with Session(self.engine) as session:
            camp_id = campaign_data["campaign_id"]
            existing = session.get(RetentionCampaign, camp_id)
            if existing:
                for k, v in campaign_data.items():
                    setattr(existing, k, v)
            else:
                camp = RetentionCampaign(**campaign_data)
                session.add(camp)
            session.commit()
            return camp_id

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Queries aggregated executive overview metrics directly from database."""
        if not self.is_connected or self.engine is None:
            return {
                "total_customers": 0,
                "historical_churn_count": 0,
                "historical_churn_rate": 0.0,
                "total_monthly_revenue": 0.0,
                "avg_tenure_months": 0.0,
                "high_risk_customers_predicted": 0,
                "average_estimated_clv": 0.0,
                "total_revenue_at_risk": 0.0,
                "latest_model_run_id": None,
                "latest_model_metrics": None,
            }
        with Session(self.engine) as session:
            total_customers = session.query(func.count(Customer.customer_id)).scalar() or 0
            
            # Snapshots
            churn_stats = session.query(
                func.sum(CustomerSnapshot.churn_status),
                func.sum(CustomerSnapshot.monthly_charges),
                func.avg(CustomerSnapshot.tenure),
            ).first()
            
            churn_count = int(churn_stats[0] or 0)
            total_monthly_rev = float(churn_stats[1] or 0.0)
            avg_tenure = float(churn_stats[2] or 0.0)
            
            # Churn predictions
            high_risk_count = session.query(func.count(ChurnPrediction.prediction_id)).filter(
                ChurnPrediction.risk_tier.in_(["High", "Critical"])
            ).scalar() or 0
            
            # Average CLV
            avg_clv = session.query(func.avg(CustomerLifetimeValue.discounted_clv)).scalar() or 0.0
            total_rev_at_risk = session.query(func.sum(CustomerLifetimeValue.revenue_at_risk)).scalar() or 0.0

            # Latest model run
            latest_run = session.query(ModelRun).order_by(ModelRun.created_at.desc()).first()

            return {
                "total_customers": total_customers,
                "historical_churn_count": churn_count,
                "historical_churn_rate": round(churn_count / total_customers, 4) if total_customers > 0 else 0.0,
                "total_monthly_revenue": round(total_monthly_rev, 2),
                "avg_tenure_months": round(avg_tenure, 1),
                "high_risk_customers_predicted": high_risk_count,
                "average_estimated_clv": round(float(avg_clv), 2),
                "total_revenue_at_risk": round(float(total_rev_at_risk), 2),
                "latest_model_run_id": latest_run.model_run_id if latest_run else None,
                "latest_model_metrics": latest_run.metrics if latest_run else None,
            }

    def get_customer_profile(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete 360-degree customer intelligence record."""
        if not self.is_connected or self.engine is None:
            return None
        with Session(self.engine) as session:
            cust = session.get(Customer, customer_id)
            if not cust:
                return None
            
            snap = session.query(CustomerSnapshot).filter(CustomerSnapshot.customer_id == customer_id).first()
            churn_pred = session.query(ChurnPrediction).filter(ChurnPrediction.customer_id == customer_id).first()
            surv_pred = session.query(SurvivalPrediction).filter(SurvivalPrediction.customer_id == customer_id).first()
            clv = session.query(CustomerLifetimeValue).filter(CustomerLifetimeValue.customer_id == customer_id).first()
            seg = session.query(CustomerSegment).filter(CustomerSegment.customer_id == customer_id).first()

            return {
                "customer_id": cust.customer_id,
                "gender": cust.gender,
                "senior_citizen": cust.senior_citizen,
                "partner": cust.partner,
                "dependents": cust.dependents,
                "contract": cust.contract,
                "payment_method": cust.payment_method,
                "internet_service": cust.internet_service,
                "phone_service": cust.phone_service,
                "tenure": snap.tenure if snap else 0,
                "monthly_charges": float(snap.monthly_charges) if snap else 0.0,
                "total_charges": float(snap.total_charges) if snap else 0.0,
                "historical_churn": snap.churn_status if snap else 0,
                "churn_probability": float(churn_pred.churn_probability) if churn_pred else None,
                "risk_tier": churn_pred.risk_tier if churn_pred else "Unscored",
                "top_churn_drivers": churn_pred.top_churn_drivers if churn_pred else None,
                "prob_active_12m": float(surv_pred.prob_active_12m) if surv_pred else None,
                "prob_active_24m": float(surv_pred.prob_active_24m) if surv_pred else None,
                "discounted_clv": float(clv.discounted_clv) if clv else None,
                "revenue_at_risk": float(clv.revenue_at_risk) if clv else None,
                "cluster_label": seg.cluster_label if seg else "Unassigned",
            }
