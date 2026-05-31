import pandas as pd
from sklearn.model_selection import train_test_split

from src import (
    get_input_data,
    cast_yes_no_variables_to_binary,
    encode_features,
    train_xgboost,
    predict_churn
)

import mlflow

def run_pipeline():
    # ── Configuration ──────────────────────────────────────────────────────────────

    DATA_PATH        = "./data/Telco_customer_churn.csv"
    TARGET_COL       = "Churn"
    DROP_COLS        = ["customerID", "TotalCharges", "PhoneService", "Churn"]
    TEST_SIZE        = 0.2
    RANDOM_STATE     = 42
    THRESHOLD        = 0.5
    EXPERIMENT_NAME  = "churn-telecom"

    XGBOOST_PARAMS = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",          # PR-AUC as early stopping metric
        "seed":             RANDOM_STATE,
        "verbosity":        0,                # suppress XGBoost training logs
        "nthread":          -1,
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Step 0: Load data ──────────────────────────────────────────────────────────

    print("── Step 0: Loading data ──────────────────────────────────────────────")

    df = get_input_data(DATA_PATH)

    # Encode target
    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=DROP_COLS)

    print(f"Dataset: {len(df):,} rows | Churn rate: {y.mean():.2%}")

    # ── Step 1: Preprocess data ──────────────────────────────────────────────────────────

    df = cast_yes_no_variables_to_binary(
        df,
        columns=['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    )

    # ── Step 2: Train / test split ─────────────────────────────────────────────────

    print("\n── Step 2: Splitting train / test ───────────────────────────────────")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(X_train):,} rows ({y_train.mean():.2%} churn)")
    print(f"Test:  {len(X_test):,} rows  ({y_test.mean():.2%} churn)")

    # ── Step 3: One-hot encoding ───────────────────────────────────────────────────

    print("\n── Step 3: Encoding features ─────────────────────────────────────────")

    X_train_enc, X_test_enc, feature_names, encoder = encode_features(
        X_train=X_train,
        X_test=X_test,
        drop="first",
    )

    print(f"Features before encoding: {X_train.shape[1]}")
    print(f"Features after encoding:  {len(feature_names)}")
    print(f"Feature names: {feature_names}")

    # ── Step 4: Train XGBoost ──────────────────────────────────────────────────────

    print("\n── Step 4: Training XGBoost ──────────────────────────────────────────")

    model, run_id = train_xgboost(
        X_train=X_train_enc,
        y_train=y_train,
        feature_names=feature_names,
        params=XGBOOST_PARAMS,
        num_boost_round=500,
        experiment_name=EXPERIMENT_NAME,
        run_name="xgboost-training",
        verbose=True,
    )

    print(f"\nMLflow run ID: {run_id}")

    # ── Step 5: Inference on test set ──────────────────────────────────────────────

    print("\n── Step 5: Generating predictions ───────────────────────────────────")

    scores_df = predict_churn(
        model=model,
        X=X_test_enc,
        feature_names=feature_names,
        threshold=THRESHOLD,
        customer_ids=X_test.reset_index(drop=True).get(
            "customerID", pd.Series(range(len(X_test)))
        ),
        monthly_charges=df.loc[X_test.index, "MonthlyCharges"].reset_index(drop=True),
        experiment_name=EXPERIMENT_NAME,
        run_name="xgboost-inference",
        verbose=True,
    )

    # ── Step 6: Output ─────────────────────────────────────────────────────────────

    print("\n── Step 6: Results ───────────────────────────────────────────────────")
    print(scores_df.head(10).to_string(index=False))
    print(f"\nRisk segment distribution:")
    print(scores_df["risk_segment"].value_counts().to_string())

    scores_df.to_csv("./data/predictions.csv", index=False)
    print("\nPredictions saved to: data/predictions.csv")
    print(f"MLflow UI: mlflow ui  →  http://localhost:5000  (experiment: {EXPERIMENT_NAME})")

if __name__ == "__main__":
    run_pipeline()