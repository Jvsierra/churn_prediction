# churn_prediction
Solution to predict customers behavior to retain them. <br />

# 1. Business Understanding <br />
**General business goal**: reduce customer churn by predicting customer behavior. <br />
**Business KPIs**: churn rate, MRR (Monthly Recurring Revenue) and LTV (Lifetime Value). <br />

# 2. Analytical Problem Definition <br />
**Target variable**: churn (categorical). <br />
**Analytical unit (granularity)**: customer. <br />
**Naive baseline**: Logistic Regression. <br />
**Business benchmark**: the current churn rate is 26.54%; the MRR is US$ 456,116.6 for the whole base, US$ 316,985.75 for the active base and US$ 139,130.85 for the lost customers; and the mean LTV is US$ 2.549,91. The model must improve those KPIs. <br />
**ML problem type**: classification. <br />
**ML metrics**: PR-AUC, ROC-AUC, precision, recall and F1-Score. <br />

Possible methods: <br />
1. Predict churn/no churn. <br />
2. Two stages method (churn/no churn prediction) then LTV. <br />
3. Survival methods. <br />
4. Probabilistical methods. <br />

