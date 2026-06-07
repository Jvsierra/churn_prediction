# Model Card — Churn Prediction Telecom B2C

## What the model does
Predicts the probability of a B2C telecom customer churning in the next 30 days.
Output is a score between 0 and 1, a binary flag, and a risk segment.

## Algorithm
XGBoost (gradient boosting). Trained on the IBM Telco Customer Churn dataset
with 7,043 customers and 20 features.

## Training
- Split: 80% train / 20% holdout
- Encoding: one-hot encoding applied inside CV loop to prevent data leakage
- Class imbalance: handled via scale_pos_weight (no SMOTE)
- Experiment tracking: MLflow

## Performance (holdout set)
- PR-AUC: 0.649
- ROC-AUC: 0.839
- Precision: 0.520
- Recall: 0.791
- F1: 0.627

## Output columns
- customer_id: customer identifier
- churn_score: probability of churning [0, 1]
- churn_pred: 1 if churn_score >= threshold (default 0.5), else 0
- risk_segment: high (>= 0.6) / medium (0.3–0.59) / low (< 0.3)

## Limitations
- Save rate (30%) and cost per action ($15) are assumed, not measured
- No A/B test — business metrics are simulated
- Model trained on a single static cohort, no temporal validation
- Threshold of 0.5 is the default — not optimized for business cost