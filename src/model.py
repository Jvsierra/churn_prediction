import re

import numpy as np
import pandas as pd

import xgboost as xgb

import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

def _sanitize_metric_name(name: str) -> str:
    """Replaces characters invalid in MLflow metric names with underscores.

    MLflow allows only alphanumerics, underscores, dashes, periods,
    spaces, and slashes. OHE feature names often contain parentheses
    (e.g. 'PaymentMethod_Credit card (automatic)') that must be removed.
    """
    return re.sub(r"[^\w\s\-./]", "_", name)


def train_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    feature_names: list[str],
    params: dict | None = None,
    num_boost_round: int = 500,
    experiment_name: str = "churn-telecom",
    run_name: str | None = None,
    verbose: bool = True,
) -> tuple[xgb.Booster, str]:
    """Trains an XGBoost model on the training set and logs the experiment to MLflow.

    Trains for a fixed number of rounds on X_train only. Logs training
    metrics, all hyperparameters, feature importances, and registers the
    model in the MLflow Model Registry.

    Args:
        X_train: Encoded training feature matrix (numpy array).
        y_train: Binary training target. 1 = churned, 0 = retained.
        feature_names: Column names matching the columns of X_train.
            Used to label DMatrix and feature importance output.
        params: XGBoost parameter dict. If None, sensible churn-oriented
            defaults are used. scale_pos_weight is computed automatically
            from y_train and should not be included.
        num_boost_round: Number of boosting rounds. Defaults to 500.
        experiment_name: MLflow experiment name. Created if it does not exist.
            Defaults to "churn-telecom".
        run_name: Optional display name for the MLflow run. Auto-generated
            by MLflow if None.
        verbose: If True, prints the training summary (run ID and all
            metrics) to stdout. If False, runs silently. Defaults to True.

    Returns:
        A tuple of two elements:
            - model (xgb.Booster): Fitted booster.
            - run_id (str): MLflow run ID for downstream reference (e.g.
              loading the model from the registry).

    Example:
        >>> model, run_id = train_xgboost(X_tr, y_tr, feature_names=names)
        >>> print(f"Run: {run_id}")
    """
    # ── Default hyperparameters ────────────────────────────────────────────────

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    default_params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "learning_rate":    0.05,
        "max_depth":        6,
        "min_child_weight": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity":        0,
        "nthread":          -1,
    }
    if params is not None:
        default_params.update(params)

    # ── DMatrix ────────────────────────────────────────────────────────────────

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

    # ── MLflow run ─────────────────────────────────────────────────────────────

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log all hyperparameters
        mlflow.log_params({
            **{k: v for k, v in default_params.items() if k != "scale_pos_weight"},
            "scale_pos_weight": round(scale_pos_weight, 4),
            "num_boost_round":  num_boost_round,
            "n_train":          len(y_train),
            "churn_rate_train": round(y_train.mean(), 4),
        })

        # ── Training ───────────────────────────────────────────────────────────

        evals_result: dict = {}

        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        # ── Evaluation metrics — train set only ────────────────────────────────

        train_proba = model.predict(dtrain)
        train_pred  = (train_proba >= 0.5).astype(int)

        train_metrics = {
            "train_pr_auc":    average_precision_score(y_train, train_proba),
            "train_roc_auc":   roc_auc_score(y_train, train_proba),
            "train_precision": precision_score(y_train, train_pred, zero_division=0),
            "train_recall":    recall_score(y_train, train_pred, zero_division=0),
            "train_f1":        f1_score(y_train, train_pred, zero_division=0),
        }

        mlflow.log_metrics(train_metrics)

        # Log per-round aucpr history on the training set
        train_aucpr_history = evals_result["train"]["aucpr"]
        for step, score in enumerate(train_aucpr_history):
            mlflow.log_metric("train_aucpr_per_round", score, step=step)

        # ── Feature importance ─────────────────────────────────────────────────

        importance_gain = model.get_score(importance_type="gain")
        for feat, score in importance_gain.items():
            safe_name = _sanitize_metric_name(f"importance_gain_{feat}")
            
            mlflow.log_metric(safe_name, score)

        # ── Register model ─────────────────────────────────────────────────────

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn-telecom-xgboost",
        )

        # Console summary — only when verbose=True
        if verbose:
            print(f"\n── Training complete ─────────────────────────")
            print(f"Run ID: {run_id}")

            for k, v in train_metrics.items():
                print(f"  {k:<22}: {v:.4f}")

    return model, run_id


def _segment(score: float) -> str:
    """Helper function to assign risk segments based on churn probability scores."""

    if score >= 0.60:
        return "high"
    elif score >= 0.30:
        return "medium"
    return "low"

def predict_churn(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.5,
    customer_ids: pd.Series | None = None,
    monthly_charges: pd.Series | None = None,
    experiment_name: str = "churn-telecom",
    run_name: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generates churn probability scores and risk segments using a trained XGBoost model.

    Logs the inference run to MLflow, including score distribution statistics,
    segment counts, and estimated MRR at risk.

    Args:
        model: Fitted xgb.Booster returned by train_xgboost.
        X: Encoded feature matrix for inference (numpy array). Must use the
            same encoding as the training data.
        feature_names: Column names matching the columns of X. Must match
            the feature_names used during training.
        threshold: Probability cutoff for classifying a customer as high risk.
            Customers above this threshold are flagged for retention action.
            Defaults to 0.5.
        customer_ids: Optional Series of customer identifiers to include in
            the output DataFrame. If None, a sequential integer index is used.
        monthly_charges: Optional Series of MonthlyCharges for each customer.
            Used to compute MRR at risk logged to MLflow. If None, MRR
            metrics are not logged.
        experiment_name: MLflow experiment name. Should match the training
            experiment. Defaults to "churn-telecom".
        run_name: Optional display name for the MLflow inference run.
        verbose: If True, prints the inference summary (customers scored,
            flagged count, segment distribution, MRR at risk) to stdout.
            If False, runs silently. Defaults to True.

    Returns:
        pd.DataFrame with the following columns:
            - customer_id: Customer identifier (from customer_ids or 0-indexed).
            - churn_score: Predicted probability of churning [0, 1].
            - churn_pred: Binary flag — 1 if churn_score >= threshold, else 0.
            - risk_segment: "high" / "medium" / "low" based on churn_score.

    Example:
        >>> scores_df = predict_churn(
        ...     model, X_inference, feature_names=names,
        ...     customer_ids=df["customerID"],
        ...     monthly_charges=df["MonthlyCharges"],
        ... )
        >>> print(scores_df.head())
    """
    # ── Inference ──────────────────────────────────────────────────────────────

    dmat        = xgb.DMatrix(X, feature_names=feature_names)
    churn_score = model.predict(dmat)
    churn_pred  = (churn_score >= threshold).astype(int)

    risk_segment = np.vectorize(_segment)(churn_score)

    output = pd.DataFrame({
        "customer_id":   customer_ids.values if customer_ids is not None
                         else np.arange(len(churn_score)),
        "churn_score":   churn_score,
        "churn_pred":    churn_pred,
        "risk_segment":  risk_segment,
    })

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or "inference"):

        mlflow.log_params({
            "threshold":   threshold,
            "n_customers": len(output),
        })

        # Score distribution
        mlflow.log_metrics({
            "score_mean":   float(churn_score.mean()),
            "score_std":    float(churn_score.std()),
            "score_min":    float(churn_score.min()),
            "score_max":    float(churn_score.max()),
            "score_median": float(np.median(churn_score)),
        })

        # Segment counts
        seg_counts = output["risk_segment"].value_counts()
        mlflow.log_metrics({
            "n_high_risk":   int(seg_counts.get("high",  0)),
            "n_medium_risk": int(seg_counts.get("medium", 0)),
            "n_low_risk":    int(seg_counts.get("low", 0)),
            "pct_high_risk": float((output["risk_segment"] == "high").mean()),
            "n_flagged":     int(churn_pred.sum()),
            "pct_flagged":   float(churn_pred.mean()),
        })

        # MRR at risk — only if MonthlyCharges provided
        if monthly_charges is not None:
            mrr_series = monthly_charges.values
            mrr_at_risk = mrr_series[churn_pred == 1].sum()
            mrr_high    = mrr_series[risk_segment == "high"].sum()
            mlflow.log_metrics({
                "mrr_at_risk":      float(mrr_at_risk),
                "mrr_high_risk":    float(mrr_high),
                "mrr_total":        float(mrr_series.sum()),
                "pct_mrr_at_risk":  float(mrr_at_risk / mrr_series.sum())
                                    if mrr_series.sum() > 0 else 0.0,
            })

        # Console summary — only when verbose=True
        if verbose:
            print(f"\n── Inference complete ────────────────────────")
            print(f"Customers scored: {len(output):,}")
            print(f"Flagged (>= {threshold}): {churn_pred.sum():,} ({churn_pred.mean():.1%})")
            print(f"Risk segments: {seg_counts.to_dict()}")
            if monthly_charges is not None:
                print(f"MRR at risk: ${mrr_at_risk:,.2f}")

    return output

def main():
    pass

if __name__ == "__main__":
    main()