"""
Feature engineering module.
Generates domain-specific indicators for telecom customer behavior:
tenure groups, billing ratios, service adoption bundles, and contract types.
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms standardized customer records by adding engineered domain features.
    Guaranteed deterministic and reproducible for training and single/batch inference.
    """
    out = df.copy()

    # 1. Tenure segmentation groups
    bins_tenure = [-1, 12, 24, 48, 60, 100]
    labels_tenure = ["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"]
    out["tenure_group"] = pd.cut(out["tenure"], bins=bins_tenure, labels=labels_tenure).astype(str)

    # 2. Monthly charge groups
    bins_mc = [0, 35, 70, 90, 500]
    labels_mc = ["Low (<$35)", "Medium ($35-$70)", "High ($70-$90)", "Very High (>$90)"]
    out["monthly_charge_group"] = pd.cut(out["monthly_charges"], bins=bins_mc, labels=labels_mc).astype(str)

    # 3. Revenue-related billing metrics
    # Historical average monthly charge: TotalCharges / (Tenure + 1)
    out["avg_monthly_charges"] = out["total_charges"] / (out["tenure"] + 1.0)
    
    # Charge ratio: Average historical charges vs Current monthly charges.
    # > 1 indicates recent price drop or promotional discount
    # < 1 indicates recent price increase or expiration of introductory discount
    out["charge_ratio"] = out["avg_monthly_charges"] / (out["monthly_charges"] + 1e-4)

    # 4. Service Adoption Indicators
    services_list = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    
    # Calculate service count based on available columns
    service_count = pd.Series(0, index=out.index)
    for col in services_list:
        if col in out.columns:
            service_count += (out[col].astype(str).str.lower() == "yes").astype(int)
    out["num_services_subscribed"] = service_count

    # Internet service indicators
    if "internet_service" in out.columns:
        out["has_internet"] = (out["internet_service"].astype(str).str.lower().isin(["dsl", "fiber optic"])).astype(int)
        out["has_fiber_optic"] = (out["internet_service"].astype(str).str.lower() == "fiber optic").astype(int)
    else:
        out["has_internet"] = 0
        out["has_fiber_optic"] = 0

    def _get_series(df: pd.DataFrame, col: str, default: Any = "") -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series(default, index=df.index)

    # Protection & Support Add-ons
    sec = (_get_series(out, "online_security").astype(str).str.lower() == "yes").astype(int)
    bck = (_get_series(out, "online_backup").astype(str).str.lower() == "yes").astype(int)
    out["has_security_or_backup"] = ((sec + bck) > 0).astype(int)
    out["has_tech_support"] = (_get_series(out, "tech_support").astype(str).str.lower() == "yes").astype(int)

    # Entertainment / Streaming Bundles
    tv = (_get_series(out, "streaming_tv").astype(str).str.lower() == "yes").astype(int)
    mov = (_get_series(out, "streaming_movies").astype(str).str.lower() == "yes").astype(int)
    out["has_streaming_bundle"] = ((tv == 1) & (mov == 1)).astype(int)

    # 5. Contract Categories & Payment Mechanics
    if "contract" in out.columns:
        out["is_month_to_month"] = (out["contract"].astype(str).str.lower() == "month-to-month").astype(int)
        out["is_long_term_contract"] = (out["contract"].astype(str).str.lower().isin(["one year", "two year"])).astype(int)
    else:
        out["is_month_to_month"] = 0
        out["is_long_term_contract"] = 0

    if "payment_method" in out.columns:
        out["is_electronic_check"] = (out["payment_method"].astype(str).str.lower() == "electronic check").astype(int)
        out["is_automatic_payment"] = (out["payment_method"].astype(str).str.lower().str.contains("automatic")).astype(int)
    else:
        out["is_electronic_check"] = 0
        out["is_automatic_payment"] = 0

    # 6. Demographics & Family Indicators
    partner_yes = (_get_series(out, "partner").astype(str).str.lower() == "yes").astype(int)
    dep_yes = (_get_series(out, "dependents").astype(str).str.lower() == "yes").astype(int)
    out["has_family"] = ((partner_yes + dep_yes) > 0).astype(int)
    
    senior = pd.to_numeric(_get_series(out, "senior_citizen", 0), errors="coerce").fillna(0).astype(int)
    out["single_senior"] = ((senior == 1) & (partner_yes == 0) & (dep_yes == 0)).astype(int)

    return out


def get_feature_metadata() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns lists of (numeric_features, categorical_features, dropped_columns).
    Ensures strict separation between predictors, identifiers, and targets.
    """
    numeric_features = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "avg_monthly_charges",
        "charge_ratio",
        "num_services_subscribed",
        "has_internet",
        "has_fiber_optic",
        "has_security_or_backup",
        "has_tech_support",
        "has_streaming_bundle",
        "is_month_to_month",
        "is_long_term_contract",
        "is_electronic_check",
        "is_automatic_payment",
        "has_family",
        "single_senior",
    ]

    categorical_features = [
        "gender",
        "partner",
        "dependents",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
        "tenure_group",
        "monthly_charge_group",
    ]

    dropped_columns = [
        "customer_id",
        "churn",
        "churn_reason",
        "city",
        "zip_code",
        "latitude",
        "longitude",
    ]

    return numeric_features, categorical_features, dropped_columns
