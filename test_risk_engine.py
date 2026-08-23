"""
test_risk_engine.py
Automated tests for the core risk engine logic.

Run with: pytest
"""

import pandas as pd
from risk_engine import evaluate_transaction


class FakeModel:
    """A fake model so tests don't need to retrain the real one every time."""
    def predict_proba(self, X):
        # Always return a fixed probability so tests are predictable
        return [[0.9, 0.1]]  # 10% fraud probability -> LOW_RISK at threshold 0.3


class FakeHighRiskModel:
    def predict_proba(self, X):
        return [[0.1, 0.9]]  # 90% fraud probability -> HIGH_RISK at threshold 0.3


def make_fake_row(columns):
    """Creates a fake transaction row with all required columns."""
    return pd.Series({col: 1.0 for col in columns})


def test_low_risk_decision():
    """A transaction with low fraud probability should be LOW_RISK."""
    columns = ['V1', 'V2', 'Amount']
    row = make_fake_row(columns)
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, FakeModel(), columns, importances, threshold=0.3)

    assert result['decision'] == 'LOW_RISK'
    assert result['status'] == 'OK'


def test_high_risk_decision():
    """A transaction with high fraud probability should be HIGH_RISK."""
    columns = ['V1', 'V2', 'Amount']
    row = make_fake_row(columns)
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, FakeHighRiskModel(), columns, importances, threshold=0.3)

    assert result['decision'] == 'HIGH_RISK'
    assert result['status'] == 'OK'


def test_missing_field_goes_to_manual_review():
    """
    Core safety guarantee: if required data is missing, the system
    must NEVER silently pass it as LOW_RISK. It must escalate instead.
    """
    columns = ['V1', 'V2', 'Amount']
    row = make_fake_row(columns).drop('V2')  # simulate missing field
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, FakeModel(), columns, importances, threshold=0.3)

    assert result['decision'] == 'MANUAL_REVIEW'
    assert result['status'] != 'OK'
    assert 'ERROR' in result['status']


def test_manual_review_never_has_a_risk_score():
    """
    If a transaction fails and goes to MANUAL_REVIEW, it should not
    also report a risk_score, since no valid prediction was made.
    """
    columns = ['V1', 'V2', 'Amount']
    row = make_fake_row(columns).drop('V1')
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, FakeModel(), columns, importances, threshold=0.3)

    assert result['decision'] == 'MANUAL_REVIEW'
    assert result['risk_score'] is None
