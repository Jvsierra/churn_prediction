# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Customer churn prediction for a B2C telecom company (IBM Telco Customer Churn dataset). An XGBoost model
scores customers by churn probability; predictions feed a Streamlit dashboard. See `README.md` for the
full EDA findings, model comparison, and business-impact simulation behind the modeling decisions.

## Commands

```bash
# Activate the local venv (Python 3.10 — see "Python version" below)
.venv\Scripts\activate.ps1        # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the full training + inference pipeline (writes data/predictions.csv)
python -m src.pipeline

# Run the Streamlit dashboard
streamlit run app/main.py

# Inspect MLflow experiment runs (tracking store is ./mlflow.db)
mlflow ui

# Run the test suite
pytest tests/ -v

# Run a single test file / class / test
pytest tests/test_model.py -v
pytest tests/test_model.py::TestSanitizeMetricName -v
pytest tests/test_model.py::TestSanitizeMetricName::test_parentheses_replaced -v
```

## Architecture

### Pipeline flow (`src/pipeline.py`)

`run_pipeline()` orchestrates the end-to-end flow, importing everything from the flat `src` namespace
(`src/__init__.py` re-exports each module's public functions — import from `src`, not the submodule,
when adding pipeline steps):

1. **Load** — `data_collection.get_input_data()` reads `data/Telco_customer_churn.csv`.
2. **Preprocess** — `data_preprocessing.cast_yes_no_variables_to_binary()` converts Yes/No columns to 1/0.
   `customerID`, `TotalCharges`, `PhoneService`, and the target `Churn` are dropped from the feature set
   before splitting: `TotalCharges` is dropped because it's structurally redundant with
   `tenure × MonthlyCharges`, and `PhoneService` because it's perfectly correlated with `MultipleLines`.
3. **Split** — stratified `train_test_split` (`random_state=42`) to preserve churn class balance.
4. **Encode** — `feature_engineering.encode_features()` one-hot encodes categorical columns, fitting the
   encoder on `X_train` only (never on test data) to avoid leakage; unseen categories at inference time
   become all-zero rows (`handle_unknown="ignore"`).
5. **Train** — `model.train_xgboost()` trains an XGBoost booster and logs params/metrics/feature
   importances to MLflow, registering the model under `churn-telecom-xgboost`.
6. **Predict** — `model.predict_churn()` scores customers, buckets them into risk segments
   (`_segment`: `>= 0.60` high, `>= 0.30` medium, else low), estimates MRR at risk, and logs an
   inference run to MLflow.
7. **Output** — writes `data/predictions.csv` (`customer_id`, `churn_score`, `churn_pred`, `risk_segment`).

### MLflow tracking

Tracking URI is a local SQLite file: `sqlite:///mlflow.db` (repo root), experiment name
`churn-telecom`. Training and inference are logged as separate runs within that experiment. Tests must
never write to this file — `tests/test_model.py` uses an `autouse` fixture (`mlflow_test_tracking`) that
redirects `mlflow.set_tracking_uri()` to a per-test temp SQLite DB; follow this pattern in any new test
that exercises `src/model.py`.

### Dashboard (`app/main.py`)

Standalone Streamlit app — it does **not** call the pipeline directly. It reads `data/predictions.csv`
and `data/Telco_customer_churn.csv` from disk (paths overridable via sidebar text inputs), merges them on
customer ID, and renders one table. Column headers are mapped through the `COLUMN_LABELS` dict in
`app/main.py` for human-readable display; add new columns there when the merged dataframe gains fields.
A chatbot feature (RAG over `app/documents/` using Ollama + FAISS) was previously built here and has been
removed — don't reintroduce it without being asked.

### Notebooks

`notebooks/` holds exploratory work (EDA, model comparison) that informed the pipeline's design choices
documented in `README.md`. They are not imported by `src/` or `app/` and aren't covered by tests.

### Python version / deployment

The project targets **Python 3.10** (the local `.venv`) — `runtime.txt` pins this for Streamlit
Community Cloud deployment. This pin exists because the packages in `requirements.txt` are version-locked
(e.g. `numpy==1.26.4`, `scikit-learn==1.4.2`) and have no prebuilt wheels on newer Python versions,
causing cloud builds to hang trying to compile from source. When bumping a pinned dependency or the
Python version, verify prebuilt wheels exist for the target combination before pushing.

`requirements.txt` should list only packages actually imported by `src/`, `app/`, `tests/`, or
`notebooks/` — it was previously generated via a global `pip freeze` and ballooned to ~250 unrelated
packages (forecasting libraries, Windows-only packages, etc.), which broke cloud deployment. Keep it
minimal, and save it as UTF-8 — a PowerShell `pip freeze > requirements.txt` on Windows can silently
write UTF-16, which breaks `pip install -r`.
