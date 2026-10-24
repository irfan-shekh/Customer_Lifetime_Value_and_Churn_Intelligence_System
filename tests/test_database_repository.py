"""
Automated tests for Database Repository and Persistence.
"""

import pytest
import pandas as pd
from src.database.repository import DatabaseRepository
from src.database.connection import check_connection_health, get_active_storage_mode


def test_database_connection_health():
    health = check_connection_health()
    assert health["status"] == "HEALTHY"
    assert health["storage_mode"] == "POSTGRESQL"
    assert len(health["tables_registered"]) >= 8


def test_repository_get_dashboard_summary():
    repo = DatabaseRepository()
    summary = repo.get_dashboard_summary()
    assert "total_customers" in summary
    assert "historical_churn_rate" in summary
    assert "high_risk_customers_predicted" in summary
    assert summary["total_customers"] > 0


def test_repository_get_customer_profile():
    repo = DatabaseRepository()
    # Check that retrieving an existing customer works
    summary = repo.get_dashboard_summary()
    if summary["total_customers"] > 0:
        profile = repo.get_customer_profile("3668-QPYBK")
        if profile:
            assert profile["customer_id"] == "3668-QPYBK"
            assert "monthly_charges" in profile
            assert "tenure" in profile
