# Churn Prediction — Telecom B2C

> **Can we identify customers about to leave — before they do?**

A full-cycle machine learning project predicting customer churn for a B2C telecom company, from raw data exploration to business impact simulation.

---

## Business Objective

> *"Retaining a customer costs 5–10× less than acquiring a new one."*

The goal was to build a model that **proactively identifies at-risk customers** so the retention team can act before churn happens — shifting the operation from reactive to predictive.

**Target KPIs:**
- Reduce monthly churn rate
- Protect MRR (Monthly Recurring Revenue)
- Maximize ROI of retention campaigns

---

## The Dataset

| Property | Value |
|---|---|
| Source | IBM Telco Customer Churn (Kaggle) |
| Customers | 7,043 |
| Features | 20 (demographic, behavioral, contractual) |
| Target | Churn — Yes / No (26% / 74%) |
| Class imbalance | Mild (26/74) |

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

### Collinearity

- `PhoneService` and `MultipleLines` are perfectly correlated by construction → dropped `PhoneService`
- `TotalCharges` ≈ `tenure × MonthlyCharges` → structurally redundant

---

## Models Tested

Five algorithms were evaluated using **stratified 5-fold cross-validation**, with PR-AUC as the primary metric (more reliable than accuracy or ROC-AUC for imbalanced classes).

| Model | Val PR-AUC | Val Recall | Val Precision | Overfitting |
|---|---|---|---|---|
| Logistic Regression | 0.65 | 0.79 | 0.51 | None |
| Random Forest | **0.67** | 0.79 | 0.52 | Minimal |
| LightGBM | 0.67 | 0.50 | **0.65** | Moderate |
| XGBoost | 0.66 | 0.77 | 0.51 | Moderate |
| SVM | 0.57 | 0.77 | 0.49 | Low |

### Why not LightGBM (highest PR-AUC)?

LightGBM achieved higher precision (65%) but only 50% recall — **it missed more than half of actual churners**. In a context where the operational cost of a retention action is low, recall matters more than precision.

---

## The Chosen Model: Random Forest

```
Best PR-AUC (excluding LightGBM)  ✓
Highest ROC-AUC                   ✓
Highest F1-Score                  ✓
Minimal train/val gap             ✓
No resampling needed              ✓
Explainable via SHAP              ✓
```

### Decision rationale

> *"When simpler and more complex models perform similarly, always prefer the simpler one."*

All models converged to similar validation performance (~0.65 PR-AUC), indicating the dataset's predictive ceiling was reached. The Random Forest offered the best trade-off between performance, stability, and interpretability.

---

## 💼 Business Impact (Holdout Simulation)

Evaluated on a **20% holdout set** never seen during training. Retention simulation: 30% save rate, $15 cost per action.

| Model | Churners Avoided | Churn Rate ↓ | MRR Retained | ROI |
|---|---|---|---|---|
| Baseline (no action) | 0 | 26.5% | $63,102 | — |
| Logistic Regression | 49 | 23.8% | $69,843 | 5.65× |
| **Random Forest** | **51** | **23.7%** | **$70,412** | **5.71×** |
| LightGBM | 28 | 25.1% | $66,178 | 6.03× |
| XGBoost | 49 | 23.8% | $69,843 | 5.65× |

**Random Forest saved the most MRR ($70.4k retained vs $63.1k baseline) with the highest number of churners avoided (51).**

LightGBM had the best ROI per action (6.03×) but fewer total churners saved — better for constrained operations; Random Forest better for maximizing total retention.

## 🧠 Explainability (SHAP)

Top drivers of churn risk, in order of importance:

1. **tenure** — shorter time as customer = highest risk
2. **Contract type** — month-to-month contracts dominate high-risk group
3. **MonthlyCharges** — higher bills with no lock-in = easier to justify leaving
4. **OnlineSecurity / TechSupport** — absence of premium services = no switching cost
5. **InternetService (Fiber)** — fiber customers are more demanding and have more alternatives

SHAP waterfall plots allow the retention team to see **exactly why** each customer was flagged — enabling personalized outreach (e.g., offer contract upgrade vs. offer security bundle).

---

## ⚠️ Assumptions & Limitations

| Item | Detail |
|---|---|
| **Save rate (30%)** | Assumed — no historical campaign data available |
| **Cost per action ($15)** | Assumed — sensitivity analysis showed rankings stable from $10–$30 |
| **No A/B test** | Business metrics are simulated, not causally validated |
| **Static dataset** | No temporal ordering — walk-forward validation not applied |
| **Single cohort** | No seasonal effects or drift analysis |
| **Threshold = 0.5** | Default threshold; business-optimal threshold not yet tuned |

---

## 🔁 What Worked vs. What Didn't

### ✅ What worked
- Stratified k-fold CV preserved the 26/74 class ratio reliably
- `class_weight="balanced_subsample"` handled imbalance without SMOTE
- Holdout set separation gave an honest estimate of business impact
- Quantile analysis + lift curves made results tangible for non-technical stakeholders

### ❌ What didn't work (and why)
- **SMOTE**: tested but didn't improve validation metrics for a 26/74 imbalance — added complexity without benefit
- **Feature selection (RFECV)**: PR-AUC plateau suggested the performance ceiling was data-driven, not feature noise
- **LightGBM as final model**: higher PR-AUC masked low recall; misleading in a context where missing churners is costly
- **Increasing model complexity**: all complex models converged to similar validation performance as Logistic Regression — no free lunch

---

## 🗂️ Project Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── Data Analysis.ipynb
│   └── Experiment - Predict_Churn or No Churn.ipynb
├── src/
│   ├── TODO: add files
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![pandas](https://img.shields.io/badge/pandas-2.x-150458)

---

## How to Run

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

## Key Takeaway

> **A well-tuned Random Forest, properly evaluated, can protect over $7,000/month in MRR that would otherwise be lost — with a 5.7× return on every dollar spent on retention.**

The most important lesson from this project: **model complexity is not the bottleneck**. The data's predictive ceiling was reached by Logistic Regression. Everything else was about choosing the right model for the business context — not chasing the highest metric.

---

*Project developed as part of a Machine Learning portfolio. Dataset: IBM Sample Data (publicly available on Kaggle).*