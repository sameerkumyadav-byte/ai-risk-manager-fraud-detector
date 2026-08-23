# ai-risk-manager-fraud-detector


**Live demo**: https://ai-risk-manager-fraud-detector-ggeywedineo9nfzg4vdk3r.streamlit.app/


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

  ## Why Random Forest, not a neural network

Fraud detection here is a tabular classification problem with ~30 
numeric features and no sequential/image/text structure — the kind 
of problem where tree-based models (Random Forest, XGBoost) reliably 
match or beat deep learning, while being faster to train, faster to 
run at inference time, and far easier to explain (via feature 
importances) to a human reviewer. A neural network would add latency 
and complexity a payment gateway can't afford, without a clear 
accuracy benefit on this kind of data. This is a deliberate choice, 
not a default.

## Results

### Threshold tradeoff (on 98 held-out fraud cases)

| Threshold | Precision | Recall | Fraud Caught | False Alarms |
|---|---|---|---|---|
| 0.2 | 0.857 | 0.857 | 84/98 | 14 |
| 0.3 | 0.922 | 0.847 | 83/98 | 7 |
| 0.5 | 0.961 | 0.745 | 73/98 | 3 |
| 0.7 | 0.971 | 0.673 | 66/98 | 2 |

### Business cost analysis (illustrative evaluation assumptions)

We assign illustrative costs to each error type — NOT real Razorpay 
pricing data — to reason about the threshold as a business decision, 
not just a statistical one:

- **False negative cost (₹122)**: average fraud amount lost when fraud 
  is missed (proxy: average fraud transaction amount in this dataset)
- **False positive cost (₹88)**: cost of wrongly blocking a legitimate 
  transaction (proxy: average normal transaction amount, representing 
  customer friction/lost sale)

| Threshold | Precision | Recall | FP | FN | FP Cost | FN Cost | Total Cost |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.857 | 0.857 | 14 | 14 | ₹1,232 | ₹1,708 | ₹2,940 |
| **0.3** | **0.922** | **0.847** | **7** | **15** | **₹616** | **₹1,830** | **₹2,446** |
| 0.5 | 0.961 | 0.745 | 3 | 25 | ₹264 | ₹3,050 | ₹3,314 |
| 0.7 | 0.971 | 0.673 | 2 | 32 | ₹176 | ₹3,904 | ₹4,080 |

**Threshold 0.3 minimizes total expected cost** under these assumptions 
— it isn't an arbitrary choice, it's the option that balances catching 
fraud against wrongly blocking legitimate customers most efficiently, 
given the stated cost proxies.

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

## Testing

Run `pytest -v` after installing dependencies. 4 tests verify core 
decision logic and the critical safety guarantee that missing/broken 
data always escalates to MANUAL_REVIEW, never silently passes as 
LOW_RISK.

## Using the app

The live dashboard has two parts:

1. **Batch Report** — runs the risk engine across 55 test transactions 
   (5 deliberately corrupted) and shows overall metrics: how many were 
   LOW_RISK, HIGH_RISK, or escalated to MANUAL_REVIEW.

2. **Transaction Analyzer** — simulates how a merchant or reviewer 
   would check a single transaction. Pick a sample (a typical 
   transaction, a known fraud case, or a corrupted one) and instantly 
   see the risk decision, risk score, and the top model signals behind 
   it. The corrupted sample demonstrates the system safely escalating 
   to `MANUAL_REVIEW` instead of guessing.

   ## AI Judgment: what uses ML, what doesn't, and why

**Where we use ML:**
A Random Forest classifier produces the fraud risk score. This is a 
tabular, structured-features problem (30 numeric columns) — exactly 
the setting where tree-based models are the standard, well-justified 
choice: fast to train, fast at inference (critical for a payment flow 
where latency matters), and explainable via feature importances.

**Where we deliberately did NOT use an LLM:**
An LLM was deliberately not used anywhere in the core decision path. 
Three concrete reasons:

1. **Latency and cost**: fraud scoring needs to happen in milliseconds 
   at payment time. An LLM call adds seconds of latency and real 
   per-call cost, for a task a lightweight classifier already solves 
   well.
2. **No natural-language reasoning is required**: the input is 30 
   numeric features, not text, transcripts, or documents. There is no 
   unstructured content for an LLM to reason over — its core 
   strength doesn't apply here.
3. **Safety**: an LLM can hallucinate or be inconsistent. A financial 
   risk decision needs to be deterministic and reproducible given the 
   same input and threshold — a property classical ML plus a fixed 
   policy threshold provides, and an LLM does not guarantee.

**Where deterministic (non-ML) logic is used:**
The decision boundary itself (`risk_score >= threshold → HIGH_RISK`) 
is plain deterministic code, not learned or inferred — thresholds are 
config, not a black box. Similarly, all failure handling (missing 
fields, corrupted input) is deterministic try/except logic, not a 
model decision — this guarantees safety-critical behavior can't 
silently change if the model changes.

**Where an LLM genuinely could add value (future work, not built here):**
Summarizing a `MANUAL_REVIEW` case in plain language for a human 
reviewer, or drafting a chargeback-evidence letter from the audit 
record. We did not build this to keep the system's core financial 
decision path deterministic, auditable, and free of hallucination risk 
within this project's scope.
