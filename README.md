# 📉 Churn Prediction — Telecom B2C

> **Can we identify customers about to leave — before they do?**

A full-cycle machine learning project predicting customer churn for a B2C telecom company, from raw data exploration to business impact simulation.

---

## 🎯 Business Objective

> *"Retaining a customer costs 5–10× less than acquiring a new one."*

The goal was to build a model that **proactively identifies at-risk customers** so the retention team can act before churn happens — shifting the operation from reactive to predictive.

**Target KPIs:**
- Reduce monthly churn rate
- Protect MRR (Monthly Recurring Revenue)
- Maximize ROI of retention campaigns

---

## 📊 The Dataset

| Property | Value |
|---|---|
| Source | IBM Telco Customer Churn (Kaggle) |
| Customers | 7,043 |
| Features | 20 (demographic, behavioral, contractual) |
| Target | Churn — Yes / No (26% / 74%) |
| Class imbalance | Mild (26/74) — no SMOTE needed |

---

## 🔍 Key Findings from EDA

### Who churns?

```
Senior citizens          → 2× higher churn rate than non-seniors
No partner / dependents  → More likely to churn (lower switching cost)
Fiber optic internet     → Higher churn (more competitive market)
Month-to-month contract  → 43% churn rate vs 3% for 2-year contracts
Electronic check payment → Highest churn among payment methods
No premium services      → No online security, backup, or tech support
Short tenure (< 12mo)    → Highest risk window for churn
```

### The core pattern

> **"Customers who find it easiest to leave — do."**

No long-term contract. No bundled services. No family plan. No friction to switching. This profile explained the majority of churners across all models.

### Collinearity findings

- `PhoneService` and `MultipleLines` are perfectly correlated by construction → dropped `PhoneService`
- `TotalCharges` ≈ `tenure × MonthlyCharges` → structurally redundant

---

## 🤖 Models Tested

Five algorithms were evaluated using **stratified 5-fold cross-validation** with one-hot encoding applied inside the CV loop (fitted only on the training fold to prevent data leakage). PR-AUC was used as the primary metric — more reliable than accuracy or ROC-AUC for imbalanced classes.

### Results

| Model | Train PR-AUC | Val PR-AUC | Val Recall | Val Precision | Val F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.647 | 0.645 | **0.795** | 0.515 | 0.625 |
| Random Forest | 0.711 | 0.653 | 0.777 | **0.532** | **0.632** |
| LightGBM | **0.758** | **0.656** | 0.774 | 0.526 | 0.626 |
| XGBoost | 0.726 | 0.649 | 0.791 | 0.520 | 0.627 |
| SVM | 0.647 | 0.579 | 0.788 | 0.509 | 0.619 |

### The leakage lesson
---

## ✅ The Chosen Model: XGBoost

```
Had similar technical metrics scores and better business KPIs scores.
```

### Decision rationale

All models converged to near-identical validation metrics. The XGBoost was chosen for its best PR-AUC and strongest business impact in the simulation.

Logistic Regression and SVM produced **zero churners avoided** in the business simulation at threshold=0.5, making them unsuitable for the operational context despite similar PR-AUC scores.

---

## 💼 Business Impact Simulation

Evaluated on the last CV fold (holdout-equivalent). Retention simulation: 30% save rate, $15 cost per action, threshold=0.5.

| Model | Churners Avoided | Churn Rate | MRR Retained | MRR Lost | LTV Total |
|---|---|---|---|---|---|
| Baseline (no action) | 0 | 26.56% | $60,906 | $27,760 | $3,050,229 |
| Logistic Regression | 0 | 26.56% | $60,906 | $27,760 | $3,050,229 |
| Random Forest | 86 | 20.45% | $67,415 | $21,251 | $3,056,738 |
| LightGBM | 86 | 20.45% | $67,436 | $21,231 | $3,056,758 |
| **XGBoost** | **87** | **20.38%** | **$67,595** | **$21,071** | **$3,056,918** |
| SVM | 0 | 26.56% | $60,906 | $27,760 | $3,050,229 |

### Delta vs. Baseline

| Model | Churners Avoided | Churn Rate ↓ | MRR Retained ↑ | LTV ↑ |
|---|---|---|---|---|
| Random Forest | 86 | −6.11pp | +$6,509 | +$6,509 |
| LightGBM | 86 | −6.11pp | +$6,529 | +$6,529 |
| **XGBoost** | **87** | **−6.18pp** | **+$6,689** | **+$6,689** |

**Random Forest, LightGBM and XGBoost are operationally equivalent** — all avoid ~86–87 churners and protect ~$6,500/month in MRR. Logistic Regression and SVM flag too few customers at this threshold to generate any retention impact. XGBoost was chosen because it was slightly better.

---

## ⚠️ Assumptions & Limitations

| Item | Detail |
|---|---|
| **Save rate (30%)** | Assumed — no historical campaign data available |
| **Cost per action ($15)** | Assumed — sensitivity analysis recommended |
| **No A/B test** | Business metrics are simulated, not causally validated |
| **Threshold = 0.5** | Default — business-optimal threshold not yet tuned |
| **Single cohort** | No seasonal effects or drift analysis |
| **LR and SVM at threshold=0.5** | Both avoid zero churners — lowering threshold would change this |

---

## 🔁 What Worked vs. What Didn't

### ✅ What worked
- Stratified k-fold CV preserved the 26/74 class ratio reliably
- `class_weight="balanced_subsample"` handled imbalance without SMOTE
- Business simulation (MRR / LTV) revealed model differences invisible in PR-AUC

### ❌ What didn't work (and why)
- **SMOTE**: tested but didn't improve validation metrics for a 26/74 imbalance
- **Feature selection**: PR-AUC plateau confirmed the ceiling was data-driven, not feature noise
- **LightGBM as final model**: moderate train/val gap; operationally equivalent to Random Forest but with more complexity
- **Logistic Regression and SVM at threshold=0.5**: zero operational impact — precision too low to flag enough true positives

---

## 🗂️ Project Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── Data Analysis.ipynb
│   └── Experiment - Predict Churn or No Churn.ipynb
├── src/
│   ├── data_collection.py
|   └── data_preprocessing.py
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![pandas](https://img.shields.io/badge/pandas-2.x-150458)
![SHAP](https://img.shields.io/badge/SHAP-0.4x-blueviolet)

---

## 🚀 How to Run

```bash
# Clone and install dependencies
git clone https://github.com/your-username/churn-prediction-telecom
cd churn-prediction-telecom
pip install -r requirements.txt

# Run EDA notebooks
jupyter notebook notebooks/

# Run full experiment
jupyter notebook notebooks/Experiment_-_Predict_Churn_or_No_Churn.ipynb
```

---

## 📌 Key Takeaways

> **XGBoost avoided 87 churners per cycle, protecting $6,700/month in MRR that would otherwise be lost.**

---

*Project developed as part of a Machine Learning portfolio. Dataset: IBM Sample Data (publicly available on Kaggle).*