"""
SQLAlchemy 2.0 ORM Schema definitions.
Maps normalized customer intelligence tables with foreign keys and check constraints.
"""

from datetime import datetime
from typing import Optional, List, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Text,
    BigInteger,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ModelRun(Base):
    __tablename__ = "model_runs"

    model_run_id = Column(String(64), primary_key=True)
    model_type = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)
    threshold = Column(Numeric(5, 4), nullable=True)
    metrics = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    churn_predictions = relationship("ChurnPrediction", back_populates="model_run")
    survival_predictions = relationship("SurvivalPrediction", back_populates="model_run")
    clv_estimates = relationship("CustomerLifetimeValue", back_populates="model_run")
    segments = relationship("CustomerSegment", back_populates="model_run")


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String(64), primary_key=True)
    gender = Column(String(16), nullable=True)
    senior_citizen = Column(Integer, default=0)
    partner = Column(String(8), nullable=True)
    dependents = Column(String(8), nullable=True)
    phone_service = Column(String(8), nullable=True)
    multiple_lines = Column(String(32), nullable=True)
    internet_service = Column(String(32), nullable=True)
    online_security = Column(String(32), nullable=True)
    online_backup = Column(String(32), nullable=True)
    device_protection = Column(String(32), nullable=True)
    tech_support = Column(String(32), nullable=True)
    streaming_tv = Column(String(32), nullable=True)
    streaming_movies = Column(String(32), nullable=True)
    contract = Column(String(32), nullable=True)
    paperless_billing = Column(String(8), nullable=True)
    payment_method = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    snapshots = relationship("CustomerSnapshot", back_populates="customer", cascade="all, delete-orphan")
    churn_predictions = relationship("ChurnPrediction", back_populates="customer", cascade="all, delete-orphan")
    survival_predictions = relationship("SurvivalPrediction", back_populates="customer", cascade="all, delete-orphan")
    clv_estimates = relationship("CustomerLifetimeValue", back_populates="customer", cascade="all, delete-orphan")
    segments = relationship("CustomerSegment", back_populates="customer", cascade="all, delete-orphan")


class CustomerSnapshot(Base):
    __tablename__ = "customer_snapshots"

    snapshot_id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    tenure = Column(Integer, nullable=False)
    monthly_charges = Column(Numeric(10, 2), nullable=False)
    total_charges = Column(Numeric(12, 2), nullable=False)
    churn_status = Column(Integer, nullable=False)
    churn_reason = Column(String(255), nullable=True)
    snapshot_date = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("customer_id", "snapshot_date", name="uq_customer_snapshot"),
    )


class ChurnPrediction(Base):
    __tablename__ = "churn_predictions"

    prediction_id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    model_run_id = Column(String(64), ForeignKey("model_runs.model_run_id"), nullable=True)
    churn_probability = Column(Numeric(5, 4), nullable=False)
    predicted_churn = Column(Integer, nullable=False)
    risk_tier = Column(String(16), nullable=False)
    top_churn_drivers = Column(JSON, nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="churn_predictions")
    model_run = relationship("ModelRun", back_populates="churn_predictions")


class SurvivalPrediction(Base):
    __tablename__ = "survival_predictions"

    survival_id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    model_run_id = Column(String(64), ForeignKey("model_runs.model_run_id"), nullable=True)
    median_remaining_tenure = Column(Numeric(6, 2), nullable=True)
    prob_active_12m = Column(Numeric(5, 4), nullable=True)
    prob_active_24m = Column(Numeric(5, 4), nullable=True)
    prob_active_36m = Column(Numeric(5, 4), nullable=True)
    prob_active_48m = Column(Numeric(5, 4), nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="survival_predictions")
    model_run = relationship("ModelRun", back_populates="survival_predictions")


class CustomerLifetimeValue(Base):
    __tablename__ = "customer_lifetime_values"

    clv_id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    model_run_id = Column(String(64), ForeignKey("model_runs.model_run_id"), nullable=True)
    forecast_horizon_months = Column(Integer, nullable=False)
    contribution_margin_pct = Column(Numeric(5, 4), nullable=False)
    annual_discount_rate = Column(Numeric(5, 4), nullable=False)
    expected_future_revenue = Column(Numeric(12, 2), nullable=False)
    expected_future_margin = Column(Numeric(12, 2), nullable=False)
    discounted_clv = Column(Numeric(12, 2), nullable=False)
    revenue_at_risk = Column(Numeric(12, 2), nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="clv_estimates")
    model_run = relationship("ModelRun", back_populates="clv_estimates")


class CustomerSegment(Base):
    __tablename__ = "customer_segments"

    segment_id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    model_run_id = Column(String(64), ForeignKey("model_runs.model_run_id"), nullable=True)
    cluster_id = Column(Integer, nullable=False)
    cluster_label = Column(String(64), nullable=False)
    cluster_description = Column(Text, nullable=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="segments")
    model_run = relationship("ModelRun", back_populates="segments")


class RetentionCampaign(Base):
    __tablename__ = "retention_campaigns"

    campaign_id = Column(String(64), primary_key=True)
    campaign_name = Column(String(128), nullable=False)
    target_risk_tier = Column(String(32), nullable=False)
    discount_pct = Column(Numeric(5, 4), nullable=False)
    discount_duration_months = Column(Integer, nullable=False)
    offer_acceptance_rate = Column(Numeric(5, 4), nullable=False)
    expected_uplift_pct = Column(Numeric(5, 4), nullable=False)
    budget_limit = Column(Numeric(12, 2), nullable=True)
    eligible_customers = Column(Integer, nullable=False)
    estimated_retained_customers = Column(Integer, nullable=False)
    campaign_cost = Column(Numeric(12, 2), nullable=False)
    retained_contribution = Column(Numeric(12, 2), nullable=False)
    net_benefit = Column(Numeric(12, 2), nullable=False)
    roi_pct = Column(Numeric(8, 2), nullable=False)
    simulated_at = Column(DateTime, default=datetime.utcnow)


class CampaignAssignment(Base):
    __tablename__ = "campaign_assignments"

    assignment_id = Column(BigInteger, primary_key=True, autoincrement=True)
    campaign_id = Column(String(64), ForeignKey("retention_campaigns.campaign_id", ondelete="CASCADE"), nullable=False)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("campaign_id", "customer_id", name="uq_campaign_assignment"),
    )
