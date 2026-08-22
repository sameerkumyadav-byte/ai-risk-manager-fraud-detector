# ai-risk-manager-fraud-detector

# AI Risk Manager — Fraud Detection Agent

An agent that detects fraudulent credit card transactions in real-time, 
explains why it flagged something, and safely hands off transactions it 
can't confidently evaluate — instead of guessing or crashing.

## Problem

Merchants lose real money to fraud that slips through, while overly 
aggressive fraud filters wrongly block legitimate customers. This agent 
detects fraud with measured precision/recall, and treats every decision 
as explainable and auditable rather than a black box.

## Approach

- **Dataset**: Kaggle Credit Card Fraud Detection dataset (284,807 
  transactions, 492 fraud cases — ~0.17%, highly imbalanced)
- **Model**: Random Forest classifier with `class_weight='balanced'` 
  to handle the extreme class imbalance
- **Train/test split**: 80/20, stratified to preserve fraud ratio in 
  both sets (394 fraud cases in training, 98 in testing)
- **Threshold selection**: tested thresholds 0.2, 0.3, 0.5, 0.7 and 
  chose 0.3 based on a cost tradeoff (see Results)

## Results

### Threshold tradeoff (on 98 held-out fraud cases)

| Threshold | Precision | Recall | Fraud Caught | False Alarms |
|---|---|---|---|---|
| 0.2 | 0.857 | 0.857 | 84/98 | 14 |
| 0.3 | 0.922 | 0.847 | 83/98 | 7 |
| 0.5 | 0.961 | 0.745 | 73/98 | 3 |
| 0.7 | 0.971 | 0.673 | 66/98 | 2 |

**Chosen threshold: 0.3** — captures nearly as much fraud as the most 
aggressive setting (0.2) while cutting false alarms on legitimate 
customers roughly in half (7 vs 14).

### Batch run (55 transactions, threshold 0.3)

- Total processed: **55**
- Resolved (no error): **50**
- Exceptions (errors): **5**
- Match rate on resolved records: **100%**

5 transactions were deliberately corrupted (missing feature data, 
simulating real-world data quality issues) to test failure handling. 
All 5 were safely routed to `MANUAL_REVIEW` instead of crashing or 
being silently misclassified.

## What broke, and how I got out

While testing the agent against transactions with missing feature 
columns (simulating corrupted or incomplete real-world data), a naive 
implementation would crash. Instead, the agent wraps every evaluation 
in a try/except block: on failure, it logs the exact error and routes 
the transaction to `MANUAL_REVIEW` rather than guessing or failing 
silently.

Example output:{'decision': 'MANUAL_REVIEW', 'fraud_probability': None,
'status': "ERROR: [...] not in index"}


This guarantees no transaction is ever silently ignored or wrongly 
auto-approved due to a data error — a broken input degrades to a safe 
human-review state, not a crash or a false pass.

## Limitations

- Dataset features (V1-V28) are anonymized via PCA, so explanations 
  reference feature names rather than real transaction attributes
- Cost estimates use the dataset's own average transaction amounts as 
  a simplified proxy, not real merchant pricing data
- In this test run, all 5 injected exceptions happened to be normal 
  (non-fraud) transactions by chance — behavior on a corrupted fraud 
  case specifically is not yet verified
- Tested on a single 55-record batch; a production system would need 
  continuous evaluation on live, streaming data

## Safety note

This project is strictly detection/defense-focused. No content here 
demonstrates how to evade fraud detection or replicate fraud patterns.
