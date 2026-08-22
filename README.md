# ai-risk-manager-fraud-detector

# AI Risk Manager — Fraud Detection Agent

An agent that detects fraudulent credit card transactions in real-time, 
explains why it flagged something, and safely hands off transactions it 
can't confidently evaluate — instead of guessing or crashing.

## Approach

- **Dataset**: Kaggle Credit Card Fraud Detection dataset (284,807 
  transactions, 492 fraud cases — ~0.17%)
- **Model**: Random Forest classifier with `class_weight='balanced'` 
  to handle the extreme class imbalance
- **Train/test split**: 80/20, stratified to preserve fraud ratio in 
  both sets
- **Threshold selection**: tested thresholds 0.2, 0.3, 0.5, 0.7 and 
  chose 0.3 based on a cost tradeoff (see Results)

  ## Results

| Threshold | Precision | Recall | Fraud Caught | False Alarms |
|---|---|---|---|---|
| 0.2 | 0.857 | 0.857 | 84/98 | 14 |
| 0.3 | 0.922 | 0.847 | 83/98 | 7 |
| 0.5 | 0.961 | 0.745 | 73/98 | 3 |
| 0.7 | 0.971 | 0.673 | 66/98 | 2 |

**Chosen threshold: 0.3** — it captures nearly as much fraud value as 
the most aggressive setting (0.2) while cutting false alarms on 
legitimate customers roughly in half.

### Batch run (60 transactions)
- Total processed: 60
- Resolved: [your number]
- Exceptions: [your number]
- Match rate on resolved records: [your %]

- ## What broke, and how I got out

While building the agent's decision function, I tested it against a 
transaction with a missing feature column (simulating corrupted or 
incomplete data, which happens in real payment systems). Instead of 
crashing, the agent catches the error, logs it, and routes the 
transaction to `MANUAL_REVIEW` status rather than silently failing 
or guessing.

Example output:{'decision': 'MANUAL_REVIEW', 'fraud_probability': None,
'status': "ERROR: [...] not found in axis"}

## Limitations

- Dataset is anonymized (V1-V28 features), so explanations reference 
  feature names, not real transaction attributes
- Cost estimates use simplified average amounts, not real merchant 
  pricing data
- Tested only on one batch of 60 transactions; a production system 
  would need continuous evaluation on live data
