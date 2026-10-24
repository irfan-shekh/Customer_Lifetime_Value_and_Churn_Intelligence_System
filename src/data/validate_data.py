"""
Data validation and data quality auditing module.
Performs schema verification, duplicate detection, missing value checks,
target leakage auditing, and data dictionary generation.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known target leakage columns in extended IBM Telco dataset
KNOWN_LEAKAGE_COLUMNS = [
    "Churn Reason",
    "Churn Score",
    "CLTV",
    "Churn Label",
]

# Non-predictive constant or identity columns
KNOWN_EXCLUDE_COLUMNS = [
    "Count",
    "Country",
    "State",
    "CustomerID",
    "customerID",
]


def validate_raw_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates the raw dataframe against expected customer intelligence schema.
    Returns structured data quality report.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "issues": [],
        "warnings": [],
        "leakage_columns_detected": [],
        "missing_summary": {},
        "target_distribution": {},
        "duplicates": 0,
        "blank_total_charges": 0,
    }

    # 1. Row count check
    if len(df) == 0:
        report["is_valid"] = False
        report["issues"].append("Dataset contains 0 rows.")
        return report

    # 2. Duplicate ID check
    id_col = None
    for candidate in ["CustomerID", "customerID"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col:
        duplicates = df[id_col].duplicated().sum()
        report["duplicates"] = int(duplicates)
        if duplicates > 0:
            report["warnings"].append(f"Found {duplicates} duplicate customer IDs.")
    else:
        report["warnings"].append("No customer ID column found.")

    # 3. Identify target column
    target_col = None
    for candidate in ["Churn Value", "Churn", "churn"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if not target_col:
        report["is_valid"] = False
        report["issues"].append("No Churn target column found.")
    else:
        target_series = df[target_col]
        # Normalize target to binary 0/1 for counting
        if target_series.dtype == object:
            norm_target = target_series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
        else:
            norm_target = target_series.fillna(0).astype(int)
        
        vc = norm_target.value_counts(normalize=True).to_dict()
        report["target_distribution"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        churn_rate = vc.get(1, 0.0)
        report["churn_rate"] = round(float(churn_rate), 4)

    # 4. Check for TotalCharges blank strings
    tc_col = None
    for candidate in ["Total Charges", "TotalCharges"]:
        if candidate in df.columns:
            tc_col = candidate
            break

    if tc_col:
        blank_mask = df[tc_col].astype(str).str.strip() == ""
        blank_count = int(blank_mask.sum())
        report["blank_total_charges"] = blank_count
        if blank_count > 0:
            report["warnings"].append(
                f"Found {blank_count} rows with blank string Total Charges (tenure=0 new customers)."
            )

    # 5. Check for target leakage columns
    for col in KNOWN_LEAKAGE_COLUMNS:
        if col in df.columns:
            report["leakage_columns_detected"].append(col)
            report["warnings"].append(
                f"Target leakage column detected: '{col}'. Must be removed before model training."
            )

    # 6. Missing value audit
    null_counts = df.isnull().sum()
    missing_dict = null_counts[null_counts > 0].to_dict()
    report["missing_summary"] = {col: int(cnt) for col, cnt in missing_dict.items()}

    logger.info(f"Validation complete: {report['row_count']} rows, {len(report['warnings'])} warnings.")
    return report


def get_data_dictionary() -> List[Dict[str, str]]:
    """Returns standardized data dictionary definitions for the Telco dataset."""
    return [
        {"column": "CustomerID", "type": "String", "category": "Identifier", "description": "Unique customer identifier (Excluded from ML features)."},
        {"column": "Gender", "type": "Categorical", "category": "Demographic", "description": "Customer gender (Male, Female)."},
        {"column": "Senior Citizen", "type": "Categorical", "category": "Demographic", "description": "Whether customer is 65 or older (Yes/No or 1/0)."},
        {"column": "Partner", "type": "Categorical", "category": "Demographic", "description": "Whether customer has a partner (Yes, No)."},
        {"column": "Dependents", "type": "Categorical", "category": "Demographic", "description": "Whether customer lives with dependents (Yes, No)."},
        {"column": "Tenure Months", "type": "Integer", "category": "Usage / Account", "description": "Number of months customer has been with the provider."},
        {"column": "Phone Service", "type": "Categorical", "category": "Service", "description": "Whether customer subscribes to home phone service."},
        {"column": "Multiple Lines", "type": "Categorical", "category": "Service", "description": "Whether customer has multiple lines (Yes, No, No phone service)."},
        {"column": "Internet Service", "type": "Categorical", "category": "Service", "description": "Internet service provider (DSL, Fiber optic, No)."},
        {"column": "Online Security", "type": "Categorical", "category": "Service", "description": "Whether customer has online security add-on."},
        {"column": "Online Backup", "type": "Categorical", "category": "Service", "description": "Whether customer has online backup add-on."},
        {"column": "Device Protection", "type": "Categorical", "category": "Service", "description": "Whether customer has equipment protection plan."},
        {"column": "Tech Support", "type": "Categorical", "category": "Service", "description": "Whether customer has premium tech support."},
        {"column": "Streaming TV", "type": "Categorical", "category": "Service", "description": "Whether customer streams television from a 3rd party."},
        {"column": "Streaming Movies", "type": "Categorical", "category": "Service", "description": "Whether customer streams movies from a 3rd party."},
        {"column": "Contract", "type": "Categorical", "category": "Contract / Billing", "description": "Contract terms: Month-to-month, One year, Two year."},
        {"column": "Paperless Billing", "type": "Categorical", "category": "Contract / Billing", "description": "Whether customer uses paperless billing (Yes, No)."},
        {"column": "Payment Method", "type": "Categorical", "category": "Contract / Billing", "description": "Payment method: Electronic check, Mailed check, Bank transfer, Credit card."},
        {"column": "Monthly Charges", "type": "Float", "category": "Financial", "description": "Amount billed to the customer monthly in USD."},
        {"column": "Total Charges", "type": "Float", "category": "Financial", "description": "Total amount billed to the customer over historical tenure in USD."},
        {"column": "Churn Value", "type": "Binary Integer", "category": "Target", "description": "1 if customer churned; 0 if customer remained active."},
        {"column": "Churn Reason", "type": "String", "category": "Target Leakage", "description": "Only populated when Churn=1. MUST BE DROPPED."},
        {"column": "Churn Score", "type": "Integer", "category": "Target Leakage", "description": "IBM pre-computed risk score (0-100). MUST BE DROPPED."},
        {"column": "CLTV", "type": "Integer", "category": "Target Leakage", "description": "IBM pre-computed CLTV index. MUST BE DROPPED."},
    ]
