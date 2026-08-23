"""
app.py
A simple dashboard for the AI Risk Manager.
Run with: streamlit run app.py
"""

import random
import streamlit as st
import pandas as pd

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
