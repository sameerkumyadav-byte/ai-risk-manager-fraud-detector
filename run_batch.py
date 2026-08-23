"""
run_batch.py
Runs the fraud detection agent across a batch of transactions
(including some deliberately corrupted ones) and prints a
summary report: match rate and unresolved exceptions.

Run with: python run_batch.py
"""

import random
import pandas as pd

from model import train_model
from risk_engine import evaluate_transaction


def run():
    print("Training model...")
    model, X_train, X_test, y_train, y_test, importances = train_model()
    print("Model trained.\n")

    # Take 55 real test transactions
    batch = X_test.iloc[:55].copy()
    true_labels = y_test.iloc[:55].copy()

    # Deliberately corrupt 5 of them to test failure handling
    random.seed(42)
    corrupt_indices = random.sample(range(len(batch)), 5)

    batch_results = []
    for idx in range(len(batch)):
        row = batch.iloc[idx]

        if idx in corrupt_indices:
            row = row.drop('V14')  # simulate missing/corrupted data

        result = evaluate_transaction(
            row, model, X_train.columns, importances, threshold=0.3
        )
        result['true_label'] = int(true_labels.iloc[idx])
        batch_results.append(result)

    results_df = pd.DataFrame(batch_results)

    total = len(results_df)
    resolved = results_df[results_df['status'] == 'OK'].copy()
    exceptions = results_df[results_df['status'] != 'OK']

    def is_match(row):
        predicted_fraud = row['decision'] == 'HIGH_RISK'
        actual_fraud = row['true_label'] == 1
        return predicted_fraud == actual_fraud

    resolved['match'] = resolved.apply(is_match, axis=1)
    match_rate = resolved['match'].mean() * 100

    print("===== BATCH REPORT =====")
    print(f"Total records processed: {total}")
    print(f"Resolved (no error): {len(resolved)}")
    print(f"Exceptions (errors): {len(exceptions)}")
    print(f"Match rate on resolved records: {match_rate:.1f}%")
    print("\nDecision breakdown:")
    print(results_df['decision'].value_counts())
    print("\nExceptions detail:")
    if len(exceptions) > 0:
        print(exceptions[['status', 'true_label']])
    else:
        print("None")


if __name__ == "__main__":
    run()
