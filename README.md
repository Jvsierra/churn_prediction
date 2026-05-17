# churn_prediction
Solution to predict customers behavior to retain them.

# 1. Business Understanding
**General business goal**: reduce customer churn by predicting customer behavior.
**Business KPIs**: churn rate, MRR (Monthly Recurring Revenue) and LTV (Lifetime Value).

# 2. Analytical Problem Definition
**Target variable**: churn (categorical).
**Analytical unit (granularity)**: customer.
**Naive baseline**: Logistic Regression.
**Business benchmark**: the current churn rate is 26.54%; the MRR is US$ 456,116.6 for the whole base, US$ 316,985.75 for the active base and US$ 139,130.85 for the lost customers; and the mean LTV is US$ 2.549,91. The model must improve those KPIs.
**ML problem type**: classification.
**ML metrics**: PR-AUC, ROC-AUC and Lift/Gain curves. After calibration, precision, recall and F1-Score.

Possible methods:
1. Predict churn/no churn.
2. Two stages method (churn/no churn prediction) then LTV.
3. Survival methods.
4. Probabilistical methods.

