"""
app.py
A simple dashboard for the AI Risk Manager.
Run with: streamlit run app.py
"""

import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

from model import train_model
from risk_engine import evaluate_transaction

st.set_page_config(page_title="AI Risk Manager", layout="wide")

st.title("🛡️ AI Risk Manager — Fraud Detection Agent")
st.caption("Defense-only fraud risk engine. When uncertain, it escalates instead of guessing.")

# Cache so the model only trains once, not on every click
@st.cache_resource
def get_model():
    return train_model()

with st.spinner("Training model (only happens once)..."):
    model, X_train, X_test, y_train, y_test, importances = get_model()

st.success("Model ready.")

if st.button("Run batch of 55 transactions (5 deliberately corrupted)"):
    batch = X_test.iloc[:55].copy()
    true_labels = y_test.iloc[:55].copy()

    random.seed(42)
    corrupt_indices = random.sample(range(len(batch)), 5)

    batch_results = []
    for idx in range(len(batch)):
        row = batch.iloc[idx]
        if idx in corrupt_indices:
            row = row.drop('V14')
        result = evaluate_transaction(row, model, X_train.columns, importances, threshold=0.3)
        result['true_label'] = int(true_labels.iloc[idx])
        batch_results.append(result)

    results_df = pd.DataFrame(batch_results)

        # Persist audit log so it's inspectable, not just in-memory
    audit_csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download audit log (CSV)",
        data=audit_csv,
        file_name="audit_log.csv",
        mime="text/csv"
    )
    
    total = len(results_df)
    resolved = results_df[results_df['status'] == 'OK']
    exceptions = results_df[results_df['status'] != 'OK']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Processed", total)
    col2.metric("Low Risk", (results_df['decision'] == 'LOW_RISK').sum())
    col3.metric("High Risk", (results_df['decision'] == 'HIGH_RISK').sum())
    col4.metric("Manual Review", (results_df['decision'] == 'MANUAL_REVIEW').sum())

    st.subheader("All Transactions")
    st.dataframe(results_df[['decision', 'risk_score', 'status', 'true_label']], use_container_width=True)

    st.subheader("⚠️ Exceptions (could not be resolved automatically)")
    if len(exceptions) > 0:
        st.dataframe(exceptions[['status', 'true_label']], use_container_width=True)
    else:
        st.write("None")

    st.subheader("Top Model Signals")
    st.write("The model relies most heavily on these anonymized features:")
    st.dataframe(importances.head(5).rename("importance"))

st.divider()
st.header("🔍 Transaction Analyzer")
st.caption("Simulates how a merchant or reviewer would check a single transaction.")

# Find one clear fraud example and one clear normal example from the test set,
# so the demo has a meaningful HIGH_RISK case to show (not just random rows)
fraud_indices = y_test[y_test == 1].index[:1]
normal_indices = y_test[y_test == 0].index[:1]

sample_options = {
    "Sample A — Typical transaction": normal_indices[0],
    "Sample B — Known fraud case (from test data)": fraud_indices[0],
    "Sample C — Corrupted transaction (missing data)": "corrupted"
}

choice = st.selectbox("Choose a transaction to analyze:", list(sample_options.keys()))

if st.button("Analyze this transaction"):
    selected = sample_options[choice]

    if selected == "corrupted":
        row = X_test.loc[normal_indices[0]].drop('V14')
        true_label = int(y_test.loc[normal_indices[0]])
    else:
        row = X_test.loc[selected]
        true_label = int(y_test.loc[selected])

    result = evaluate_transaction(row, model, X_train.columns, importances, threshold=0.3)

    st.subheader("Result")
    decision_color = {
        "LOW_RISK": "🟢",
        "HIGH_RISK": "🔴",
        "MANUAL_REVIEW": "🟡"
    }
    st.markdown(f"### {decision_color.get(result['decision'], '')} {result['decision']}")

    col1, col2 = st.columns(2)
    col1.metric("Risk Score", result['risk_score'] if result['risk_score'] is not None else "N/A")
    col2.metric("Threshold Used", result['threshold_used'])

    if result['status'] == 'OK':
        st.write("**Top contributing signals:**")
        st.json(result['top_signals'])
    else:
        st.error(f"System could not confidently evaluate this transaction.\n\nReason: {result['status']}")
        st.write("**This is why it was safely escalated to MANUAL_REVIEW instead of guessing.**")

    st.caption(f"(For reference — actual label in dataset: {'FRAUD' if true_label == 1 else 'NORMAL'})")

st.divider()
st.header("📊 Model Performance")
st.caption("Measured on the held-out test set (never seen during training).")

y_test_probs = model.predict_proba(X_test)[:, 1]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision-Recall Curve")
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_test_probs)

    fig1, ax1 = plt.subplots()
    ax1.plot(recall_vals, precision_vals, color='#2ecc71')
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision vs Recall (all thresholds)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.subheader("Confusion Matrix (threshold = 0.3)")
    y_pred_at_threshold = (y_test_probs >= 0.3).astype(int)
    cm = confusion_matrix(y_test, y_pred_at_threshold)

    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    st.pyplot(fig2)

st.caption(
    "The precision-recall curve shows the full tradeoff across all possible "
    "thresholds. The confusion matrix shows the specific outcome at our "
    "chosen threshold of 0.3, on the full held-out test set."
)
