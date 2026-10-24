"""
Dataset loading and column standardization module.
Loads raw Telco Churn CSV data and standardizes column naming across variations.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import logging

from src.config import RAW_DATA_PATH, FALLBACK_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


COLUMN_MAPPING = {
    # Identifiers
    "CustomerID": "customer_id",
    "customerID": "customer_id",
    
    # Demographics
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    
    # Usage / Services
    "Tenure Months": "tenure",
    "tenure": "tenure",
    "Phone Service": "phone_service",
    "PhoneService": "phone_service",
    "Multiple Lines": "multiple_lines",
    "MultipleLines": "multiple_lines",
    "Internet Service": "internet_service",
    "InternetService": "internet_service",
    "Online Security": "online_security",
    "OnlineSecurity": "online_security",
    "Online Backup": "online_backup",
    "OnlineBackup": "online_backup",
    "Device Protection": "device_protection",
    "DeviceProtection": "device_protection",
    "Tech Support": "tech_support",
    "TechSupport": "tech_support",
    "Streaming TV": "streaming_tv",
    "StreamingTV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "StreamingMovies": "streaming_movies",
    
    # Contract & Billing
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "PaperlessBilling": "paperless_billing",
    "Payment Method": "payment_method",
    "PaymentMethod": "payment_method",
    "Monthly Charges": "monthly_charges",
    "MonthlyCharges": "monthly_charges",
    "Total Charges": "total_charges",
    "TotalCharges": "total_charges",
    
    # Target
    "Churn Value": "churn",
    "Churn": "churn",
    
    # Optional context (not for training)
    "Churn Reason": "churn_reason",
    "City": "city",
    "Zip Code": "zip_code",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads raw dataset from specified path or default workspace paths."""
    target_path = filepath or RAW_DATA_PATH
    if not Path(target_path).exists():
        if Path(FALLBACK_DATA_PATH).exists():
            target_path = FALLBACK_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Telco Customer Churn dataset not found at {target_path} or {FALLBACK_DATA_PATH}. "
                "Please place 'Telco_customer_churn.csv' in the project root or data/raw/ directory."
            )
    
    logger.info(f"Loading raw dataset from {target_path}")
    df = pd.read_csv(target_path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and types to canonical schema.
    Safely parses numeric columns (including blank string handling in total_charges).
    """
    df_clean = df.copy()
    
    # Rename known columns
    rename_dict = {col: COLUMN_MAPPING[col] for col in df_clean.columns if col in COLUMN_MAPPING}
    df_clean = df_clean.rename(columns=rename_dict)
    
    # Ensure customer_id exists
    if "customer_id" not in df_clean.columns:
        df_clean["customer_id"] = [f"CUST-{i:05d}" for i in range(len(df_clean))]
        
    # Standardize target 'churn'
    if "churn" in df_clean.columns:
        if df_clean["churn"].dtype == object:
            df_clean["churn"] = (
                df_clean["churn"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            df_clean["churn"] = df_clean["churn"].fillna(0).astype(int)

    # Standardize senior_citizen
    if "senior_citizen" in df_clean.columns:
        if df_clean["senior_citizen"].dtype == object:
            df_clean["senior_citizen"] = (
                df_clean["senior_citizen"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            df_clean["senior_citizen"] = df_clean["senior_citizen"].fillna(0).astype(int)

    # Standardize numeric columns: tenure, monthly_charges, total_charges
    if "tenure" in df_clean.columns:
        df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce").fillna(0).astype(int)

    if "monthly_charges" in df_clean.columns:
        df_clean["monthly_charges"] = pd.to_numeric(df_clean["monthly_charges"], errors="coerce").fillna(0.0)

    if "total_charges" in df_clean.columns:
        # Handle blank strings: when tenure is 0, total_charges is 0.0
        df_clean["total_charges"] = pd.to_numeric(
            df_clean["total_charges"].astype(str).str.strip(), errors="coerce"
        )
        # Impute with 0.0 for tenure 0, or tenure * monthly_charges
        missing_tc = df_clean["total_charges"].isna()
        if missing_tc.any():
            df_clean.loc[missing_tc & (df_clean["tenure"] == 0), "total_charges"] = 0.0
            remaining_missing = df_clean["total_charges"].isna()
            if remaining_missing.any():
                df_clean.loc[remaining_missing, "total_charges"] = (
                    df_clean.loc[remaining_missing, "tenure"] * df_clean.loc[remaining_missing, "monthly_charges"]
                )

    return df_clean
