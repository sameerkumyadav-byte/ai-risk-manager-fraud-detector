"""
risk_engine.py
Takes a single transaction, scores it using the trained model,
and returns a risk decision: LOW_RISK, HIGH_RISK, or MANUAL_REVIEW.

Design principle: when the system is uncertain or the input is
broken, it escalates to MANUAL_REVIEW instead of guessing.
"""

import datetime
import pandas as pd

# This log stores every decision made — the audit trail
audit_log = []


def evaluate_transaction(transaction_row, model, feature_columns, importances, threshold=0.3):
    """
    transaction_row: a single row of transaction data (pandas Series)
    model: the trained RandomForestClassifier
    feature_columns: the list/index of columns the model expects (X_train.columns)
    importances: feature importance Series from the trained model
    threshold: risk score cutoff for HIGH_RISK vs LOW_RISK
    """
    try:
        features = pd.DataFrame([transaction_row])[feature_columns]
        prob = model.predict_proba(features)[0][1]

        decision = "HIGH_RISK" if prob >= threshold else "LOW_RISK"

        top_features = importances.head(3).index.tolist()
        reason = {feat: float(transaction_row[feat]) for feat in top_features}

        result = {
            "timestamp": str(datetime.datetime.now()),
            "decision": decision,
            "risk_score": round(float(prob), 4),
            "threshold_used": threshold,
            "top_signals": reason,
            "status": "OK"
        }

    except Exception as e:
        # Graceful failure: broken/missing input never becomes LOW_RISK.
        # It is escalated for human review instead.
        result = {
            "timestamp": str(datetime.datetime.now()),
            "decision": "MANUAL_REVIEW",
            "risk_score": None,
            "threshold_used": threshold,
            "top_signals": None,
            "status": f"ERROR: {str(e)}"
        }

    audit_log.append(result)
    return result
