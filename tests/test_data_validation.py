"""
Automated tests for data loading, validation, and standardization.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.validate_data import validate_raw_dataframe, get_data_dictionary
from src.data.load_data import standardize_columns


def test_validate_raw_dataframe_valid():
    df = pd.DataFrame({
        "CustomerID": ["C-1", "C-2", "C-3"],
        "Tenure Months": [12, 24, 0],
        "Monthly Charges": [50.0, 75.0, 20.0],
        "Total Charges": ["600.0", "1800.0", " "],  # Blank string test
        "Churn Value": [0, 1, 0],
        "Churn Reason": [np.nan, "Competitor offer", np.nan],
    })
    report = validate_raw_dataframe(df)
    assert report["is_valid"] is True
    assert report["row_count"] == 3
    assert report["blank_total_charges"] == 1
    assert "Churn Reason" in report["leakage_columns_detected"]


def test_validate_raw_dataframe_empty():
    df = pd.DataFrame()
    report = validate_raw_dataframe(df)
    assert report["is_valid"] is False
    assert len(report["issues"]) > 0


def test_standardize_columns_blank_imputation():
    df = pd.DataFrame({
        "CustomerID": ["C-1", "C-2"],
        "Tenure Months": [0, 10],
        "Monthly Charges": [50.0, 60.0],
        "Total Charges": [" ", "600.0"],
        "Churn Value": [0, 1],
    })
    clean = standardize_columns(df)
    assert "customer_id" in clean.columns
    assert "tenure" in clean.columns
    assert "total_charges" in clean.columns
    assert clean.loc[0, "total_charges"] == 0.0  # Imputed for tenure 0
    assert clean.loc[1, "total_charges"] == 600.0


def test_data_dictionary_structure():
    d = get_data_dictionary()
    assert isinstance(d, list)
    assert len(d) > 15
    for item in d:
        assert "column" in item
        assert "category" in item
