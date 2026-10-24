"""
Automated tests for data preprocessing and stratified splitting.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocess import split_data, DataPreprocessor
from src.data.features import engineer_features


def test_split_data_proportions():
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"C-{i}" for i in range(n)],
        "churn": [1 if i < 30 else 0 for i in range(n)],
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 100, n),
        "total_charges": np.random.uniform(100, 3000, n),
    })
    train, val, test = split_data(df, test_size=0.20, val_size=0.20, random_state=42)
    assert len(test) == 20
    assert len(train) + len(val) + len(test) == n
    # Verify stratification
    assert abs(train["churn"].mean() - 0.30) < 0.05
    assert abs(test["churn"].mean() - 0.30) < 0.05


def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "customer_id": ["C-1", "C-2", "C-3"],
        "gender": ["Male", "Female", "Male"],
        "senior_citizen": [0, 1, 0],
        "partner": ["Yes", "No", "Yes"],
        "dependents": ["No", "No", "Yes"],
        "tenure": [12, 24, 36],
        "phone_service": ["Yes", "No", "Yes"],
        "multiple_lines": ["No", "No phone service", "Yes"],
        "internet_service": ["DSL", "Fiber optic", "No"],
        "online_security": ["Yes", "No", "No internet service"],
        "online_backup": ["No", "Yes", "No internet service"],
        "device_protection": ["Yes", "No", "No internet service"],
        "tech_support": ["No", "No", "No internet service"],
        "streaming_tv": ["No", "Yes", "No internet service"],
        "streaming_movies": ["No", "Yes", "No internet service"],
        "contract": ["Month-to-month", "One year", "Two year"],
        "paperless_billing": ["Yes", "No", "Yes"],
        "payment_method": ["Electronic check", "Mailed check", "Bank transfer"],
        "monthly_charges": [55.0, 85.0, 25.0],
        "total_charges": [660.0, 2040.0, 900.0],
        "churn": [0, 1, 0],
    })
    pre = DataPreprocessor()
    X_tr, feature_names = pre.fit_transform(df)
    assert X_tr.shape[0] == 3
    assert len(feature_names) == X_tr.shape[1]
    assert pre.is_fitted is True

    # Transform new data
    X_new, _ = pre.transform(df.iloc[:1])
    assert X_new.shape[0] == 1
    assert X_new.shape[1] == X_tr.shape[1]
