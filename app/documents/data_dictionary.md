# Data Dictionary — Telco Customer Churn Dataset

## Source
IBM Sample Dataset — publicly available on Kaggle.

## Target variable
- Churn: whether the customer left in the last month (Yes / No → 1 / 0)

## Features used in the model

### Demographic
- SeniorCitizen: 1 if the customer is a senior citizen, 0 otherwise
- Partner: whether the customer has a partner (Yes / No → 1 / 0)
- Dependents: whether the customer has dependents (Yes / No → 1 / 0)

### Account
- tenure: number of months the customer has been with the company
- Contract: contract term (Month-to-month / One year / Two year)
- PaperlessBilling: whether the customer uses paperless billing (Yes / No → 1 / 0)
- PaymentMethod: Electronic check / Mailed check / Bank transfer / Credit card
- MonthlyCharges: amount charged monthly in USD

### Services
- MultipleLines: whether the customer has multiple phone lines
- InternetService: DSL / Fiber optic / No
- OnlineSecurity: whether the customer has online security service
- OnlineBackup: whether the customer has online backup service
- DeviceProtection: whether the customer has device protection
- TechSupport: whether the customer has tech support
- StreamingTV: whether the customer streams TV
- StreamingMovies: whether the customer streams movies

## Columns dropped before training
- customerID: identifier, no predictive value
- TotalCharges: structurally derived from tenure × MonthlyCharges
- PhoneService: perfectly correlated with MultipleLines (data leakage risk)
- Churn: the target variable itself