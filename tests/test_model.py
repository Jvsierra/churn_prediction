import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow

from src.model import train_xgboost, predict_churn, _sanitize_metric_name, _segment


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mlflow_test_tracking(tmp_path):
    """Redirect all MLflow tracking to a temp SQLite DB for isolation."""
    db_path = tmp_path / "mlflow_test.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    mlflow.set_tracking_uri("")


@pytest.fixture
def binary_dataset():
    """Minimal imbalanced binary dataset (75/25 split) for churn."""
    rng = np.random.default_rng(42)
    n   = 200
    X   = rng.standard_normal((n, 5)).astype(np.float32)
    y   = pd.Series((rng.random(n) < 0.25).astype(int))
    feature_names = [f"feature_{i}" for i in range(5)]
    return X, y, feature_names


@pytest.fixture
def trained_model(binary_dataset):
    """Returns a fitted XGBoost booster trained on the binary dataset."""
    X, y, feature_names = binary_dataset
    model, run_id = train_xgboost(
        X_train=X,
        y_train=y,
        feature_names=feature_names,
        num_boost_round=10,   # fast for testing
        verbose=False,
    )
    return model, run_id, X, y, feature_names


# ── Tests: _sanitize_metric_name ──────────────────────────────────────────────

class TestSanitizeMetricName:

    def test_parentheses_replaced(self):
        """Parentheses must be replaced with underscores."""
        result = _sanitize_metric_name("importance_gain_Payment (auto)")
        assert "(" not in result
        assert ")" not in result

    def test_valid_chars_unchanged(self):
        """Alphanumerics, underscores, dashes, periods, spaces and slashes must pass."""
        name = "importance_gain_tenure"
        assert _sanitize_metric_name(name) == name

    def test_returns_string(self):
        """Must return a string regardless of input."""
        assert isinstance(_sanitize_metric_name("any (value)"), str)


# ── Tests: _segment ───────────────────────────────────────────────────────────

class TestSegment:

    def test_high_risk_above_060(self):
        assert _segment(0.60) == "high"
        assert _segment(0.95) == "high"

    def test_medium_risk_between_030_and_060(self):
        assert _segment(0.30) == "medium"
        assert _segment(0.59) == "medium"

    def test_low_risk_below_030(self):
        assert _segment(0.00) == "low"
        assert _segment(0.29) == "low"


# ── Tests: train_xgboost ──────────────────────────────────────────────────────

class TestTrainXgboost:

    def test_returns_booster(self, trained_model):
        """train_xgboost must return an xgb.Booster."""
        model, _, _, _, _ = trained_model
        assert isinstance(model, xgb.Booster)

    def test_returns_run_id_string(self, trained_model):
        """train_xgboost must return a non-empty run_id string."""
        _, run_id, _, _, _ = trained_model
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_mlflow_run_logged(self, trained_model):
        """The MLflow run must be findable by run_id."""
        _, run_id, _, _, _ = trained_model
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run is not None

    def test_train_metrics_logged(self, trained_model):
        """Core training metrics must be present in the MLflow run."""
        _, run_id, _, _, _ = trained_model
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        for key in ["train_pr_auc", "train_roc_auc", "train_precision",
                    "train_recall", "train_f1"]:
            assert key in metrics, f"Expected metric '{key}' not found in MLflow run."

    def test_hyperparams_logged(self, trained_model):
        """num_boost_round and n_train must be logged as parameters."""
        _, run_id, _, _, _ = trained_model
        client = mlflow.tracking.MlflowClient()
        params = client.get_run(run_id).data.params
        assert "num_boost_round" in params
        assert "n_train" in params

    def test_custom_params_applied(self, binary_dataset):
        """Custom params dict must override defaults."""
        X, y, feature_names = binary_dataset
        custom = {"learning_rate": 0.01, "max_depth": 3}
        model, run_id = train_xgboost(
            X, y, feature_names,
            params=custom,
            num_boost_round=5,
            verbose=False,
        )
        client = mlflow.tracking.MlflowClient()
        params = client.get_run(run_id).data.params
        assert params["learning_rate"] == "0.01"
        assert params["max_depth"] == "3"

    def test_scale_pos_weight_computed_from_y(self, binary_dataset):
        """scale_pos_weight must be derived from y_train class balance."""
        X, y, feature_names = binary_dataset
        expected = round((y == 0).sum() / (y == 1).sum(), 4)
        _, run_id = train_xgboost(X, y, feature_names, num_boost_round=5, verbose=False)
        client = mlflow.tracking.MlflowClient()
        logged = float(client.get_run(run_id).data.params["scale_pos_weight"])
        assert abs(logged - expected) < 1e-3

    def test_verbose_false_produces_no_output(self, binary_dataset, capsys):
        """verbose=False must produce no stdout output."""
        X, y, feature_names = binary_dataset
        train_xgboost(X, y, feature_names, num_boost_round=5, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_produces_output(self, binary_dataset, capsys):
        """verbose=True must print a training summary to stdout."""
        X, y, feature_names = binary_dataset
        train_xgboost(X, y, feature_names, num_boost_round=5, verbose=True)
        captured = capsys.readouterr()
        assert "Training complete" in captured.out


# ── Tests: predict_churn ──────────────────────────────────────────────────────

class TestPredictChurn:

    def test_returns_dataframe(self, trained_model):
        """predict_churn must return a pandas DataFrame."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, trained_model):
        """Output DataFrame must have the four expected columns."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert set(result.columns) == {"customer_id", "churn_score",
                                       "churn_pred", "risk_segment"}

    def test_row_count_matches_input(self, trained_model):
        """Output row count must match the number of input samples."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert len(result) == len(X)

    def test_churn_score_in_range(self, trained_model):
        """All churn_score values must be in [0, 1]."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert result["churn_score"].between(0, 1).all()

    def test_churn_pred_is_binary(self, trained_model):
        """churn_pred must contain only 0 and 1."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert set(result["churn_pred"].unique()).issubset({0, 1})

    def test_risk_segment_values(self, trained_model):
        """risk_segment must only contain 'high', 'medium', or 'low'."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names, verbose=False)
        assert set(result["risk_segment"].unique()).issubset({"high", "medium", "low"})

    def test_threshold_respected(self, trained_model):
        """churn_pred must be 1 iff churn_score >= threshold."""
        model, _, X, y, feature_names = trained_model
        threshold = 0.3
        result = predict_churn(model, X, feature_names,
                               threshold=threshold, verbose=False)
        expected_pred = (result["churn_score"] >= threshold).astype(int)
        pd.testing.assert_series_equal(
            result["churn_pred"].reset_index(drop=True),
            expected_pred.reset_index(drop=True),
            check_names=False,
        )

    def test_customer_ids_used_when_provided(self, trained_model):
        """customer_id column must use the provided Series values."""
        model, _, X, y, feature_names = trained_model
        ids = pd.Series([f"ID-{i}" for i in range(len(X))])
        result = predict_churn(model, X, feature_names,
                               customer_ids=ids, verbose=False)
        assert list(result["customer_id"]) == list(ids)

    def test_customer_ids_sequential_when_none(self, trained_model):
        """customer_id must be a sequential integer index when not provided."""
        model, _, X, y, feature_names = trained_model
        result = predict_churn(model, X, feature_names,
                               customer_ids=None, verbose=False)
        expected = list(range(len(X)))
        assert list(result["customer_id"]) == expected

    def test_mlflow_inference_run_logged(self, trained_model):
        """predict_churn must create an MLflow run with score metrics."""
        model, _, X, y, feature_names = trained_model
        predict_churn(model, X, feature_names, verbose=False,
                      run_name="test-inference")
        client = mlflow.tracking.MlflowClient()
        runs   = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("churn-telecom").experiment_id],
            filter_string="tags.mlflow.runName = 'test-inference'",
        )
        assert len(runs) == 1
        metrics = runs[0].data.metrics
        assert "score_mean" in metrics
        assert "n_flagged" in metrics

    def test_mrr_metrics_logged_when_charges_provided(self, trained_model):
        """MRR metrics must be logged when monthly_charges is provided."""
        model, _, X, y, feature_names = trained_model
        charges = pd.Series(np.random.uniform(30, 120, len(X)))
        predict_churn(model, X, feature_names,
                      monthly_charges=charges,
                      run_name="test-mrr",
                      verbose=False)
        client = mlflow.tracking.MlflowClient()
        runs   = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("churn-telecom").experiment_id],
            filter_string="tags.mlflow.runName = 'test-mrr'",
        )
        metrics = runs[0].data.metrics
        assert "mrr_at_risk" in metrics
        assert "mrr_total" in metrics

    def test_verbose_false_produces_no_output(self, trained_model, capsys):
        """verbose=False must produce no stdout output."""
        model, _, X, y, feature_names = trained_model
        predict_churn(model, X, feature_names, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_produces_output(self, trained_model, capsys):
        """verbose=True must print an inference summary to stdout."""
        model, _, X, y, feature_names = trained_model
        predict_churn(model, X, feature_names, verbose=True)
        captured = capsys.readouterr()
        assert "Inference complete" in captured.out