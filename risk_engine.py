"""
risk_engine.py
Takes a single transaction, scores it using the trained model,
and returns a risk decision: LOW_RISK, HIGH_RISK, or MANUAL_REVIEW.

Uses SHAP to explain WHY a specific transaction was scored the way
it was, rather than only showing generic feature importance.

Design principle: when the system is uncertain or the input is
broken, it escalates to MANUAL_REVIEW instead of guessing.
"""

import datetime
import pandas as pd
import shap

MODEL_VERSION = "fraud-rf-v1"
audit_log = []


def get_shap_explanation(model, features_df, feature_names, top_n=3):
    """
    Returns the top_n features that most influenced THIS specific
    prediction, with signed SHAP values (positive = pushed toward
    fraud, negative = pushed toward normal).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    fraud_contributions = shap_values[0][:, 1]
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': fraud_contributions
    })
    shap_df['abs_value'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_value', ascending=False).head(top_n)

    return {
        row['feature']: round(float(row['shap_value']), 5)
        for _, row in shap_df.iterrows()
    }


def evaluate_transaction(transaction_row, model, feature_columns, importances, threshold=0.3):
    """
    transaction_row: a single row of transaction data (pandas Series)
    model: the trained RandomForestClassifier
    feature_columns: the list/index of columns the model expects
    importances: feature importance Series (kept as fallback reference)
    threshold: risk score cutoff for HIGH_RISK vs LOW_RISK
    """
    try:
        features = pd.DataFrame([transaction_row])[feature_columns]
        prob = model.predict_proba(features)[0][1]

        decision = "HIGH_RISK" if prob >= threshold else "LOW_RISK"

        top_signals = get_shap_explanation(model, features, feature_columns)

        result = {
            "timestamp": str(datetime.datetime.now()),
            "decision": decision,
            "risk_score": round(float(prob), 4),
            "threshold_used": threshold,
            "top_signals": top_signals,
            "status": "OK",
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        result = {
            "timestamp": str(datetime.datetime.now()),
            "decision": "MANUAL_REVIEW",
            "risk_score": None,
            "threshold_used": threshold,
            "top_signals": None,
            "status": f"ERROR: {str(e)}",
            "model_version": MODEL_VERSION
        }

    audit_log.append(result)
    return result
