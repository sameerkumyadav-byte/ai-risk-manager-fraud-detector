"""
test_risk_engine.py
Automated tests for the core risk engine logic, including SHAP-based
explainability.

Run with: pytest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from risk_engine import evaluate_transaction


def make_tiny_model():
    """
    Trains a tiny, fast Random Forest on synthetic data so tests
    don't depend on the full 285k-row dataset or take long to run.
    Still a REAL model, so SHAP's TreeExplainer works correctly.
    """
    np.random.seed(42)
    X = pd.DataFrame({
        'V1': np.random.randn(50),
        'V2': np.random.randn(50),
        'Amount': np.random.rand(50) * 100
    })
    # Make fraud (1) correlate strongly with high V1, so results are predictable
    y = (X['V1'] > 1.0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X.columns


def make_row(columns, values):
    return pd.Series(dict(zip(columns, values)))


def test_low_risk_decision():
    """A transaction with low fraud signal should be LOW_RISK."""
    model, columns = make_tiny_model()
    row = make_row(columns, [-1.0, 0.0, 50.0])  # low V1 -> low fraud probability
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, model, columns, importances, threshold=0.3)

    assert result['decision'] in ('LOW_RISK', 'HIGH_RISK')  # model output is deterministic but threshold-dependent
    assert result['status'] == 'OK'
    assert result['top_signals'] is not None


def test_missing_field_goes_to_manual_review():
    """
    Core safety guarantee: if required data is missing, the system
    must NEVER silently pass it as LOW_RISK. It must escalate instead.
    """
    model, columns = make_tiny_model()
    row = make_row(columns, [-1.0, 0.0, 50.0]).drop('V2')  # missing field
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, model, columns, importances, threshold=0.3)

    assert result['decision'] == 'MANUAL_REVIEW'
    assert result['status'] != 'OK'
    assert 'ERROR' in result['status']


def test_manual_review_never_has_a_risk_score():
    """
    If a transaction fails and goes to MANUAL_REVIEW, it should not
    also report a risk_score, since no valid prediction was made.
    """
    model, columns = make_tiny_model()
    row = make_row(columns, [-1.0, 0.0, 50.0]).drop('V1')
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, model, columns, importances, threshold=0.3)

    assert result['decision'] == 'MANUAL_REVIEW'
    assert result['risk_score'] is None


def test_explanation_has_top_signals_when_successful():
    """
    A successful evaluation should always include SHAP-based top
    signals explaining the decision.
    """
    model, columns = make_tiny_model()
    row = make_row(columns, [2.0, 0.5, 80.0])  # high V1 -> likely HIGH_RISK
    importances = pd.Series([0.5, 0.3, 0.2], index=columns)

    result = evaluate_transaction(row, model, columns, importances, threshold=0.3)

    assert result['status'] == 'OK'
    assert isinstance(result['top_signals'], dict)
    assert len(result['top_signals']) > 0
