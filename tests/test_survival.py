"""
Automated tests for Survival Analysis with Lifelines.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.survival import SurvivalAnalysisService


def test_survival_service_load_and_predict():
    service = SurvivalAnalysisService.load()
    assert service.kmf is not None

    table = service.get_survival_table()
    assert not table.empty
    assert "survival_probability" in table.columns
    # Survival probability is monotonic non-increasing
    probs = table["survival_probability"].values
    assert np.all(np.diff(probs) <= 1e-4)


def test_conditional_survival_bounds():
    service = SurvivalAnalysisService.load()
    cond = service.predict_conditional_survival(current_tenure=12, future_horizons=[12, 24, 36])
    assert 12 in cond
    assert 24 in cond
    assert 36 in cond
    for h, p in cond.items():
        assert 0.0 <= p <= 1.0
    # Later horizon must have smaller or equal survival probability
    assert cond[12] >= cond[24] >= cond[36]
