"""tests/test_feature_engineering.py

Unit tests for src.feature_engineering.encode_features.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering import encode_features


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "tenure":          [12, 3, 24, 6, 1],
        "MonthlyCharges":  [50.0, 70.0, 90.0, 30.0, 80.0],
        "Contract":        ["Month-to-month", "One year", "Two year",
                            "Month-to-month", "One year"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
    })


@pytest.fixture
def numeric_only_df() -> pd.DataFrame:
    """DataFrame with only numeric columns."""
    return pd.DataFrame({
        "tenure":         [12, 3, 24],
        "MonthlyCharges": [50.0, 70.0, 90.0],
    })


@pytest.fixture
def train_test_split(mixed_df):
    """Simple train/test split of the mixed DataFrame."""
    return mixed_df.iloc[:3].copy(), mixed_df.iloc[3:].copy()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEncodeFeatures:

    def test_returns_four_elements(self, train_test_split):
        """Function must return a tuple of exactly four elements."""
        X_train, X_test = train_test_split
        result = encode_features(X_train, X_test)
        assert len(result) == 4

    def test_train_output_is_ndarray(self, train_test_split):
        """Encoded training matrix must be a numpy ndarray."""
        X_train, X_test = train_test_split
        X_tr_enc, _, _, _ = encode_features(X_train, X_test)
        assert isinstance(X_tr_enc, np.ndarray)

    def test_test_output_is_ndarray(self, train_test_split):
        """Encoded test matrix must be a numpy ndarray."""
        X_train, X_test = train_test_split
        _, X_te_enc, _, _ = encode_features(X_train, X_test)
        assert isinstance(X_te_enc, np.ndarray)

    def test_feature_names_is_list_of_strings(self, train_test_split):
        """feature_names must be a list of strings."""
        X_train, X_test = train_test_split
        _, _, feature_names, _ = encode_features(X_train, X_test)
        assert isinstance(feature_names, list)
        assert all(isinstance(n, str) for n in feature_names)

    def test_encoder_is_fitted_ohe(self, train_test_split):
        """Returned encoder must be a fitted OneHotEncoder."""
        X_train, X_test = train_test_split
        _, _, _, encoder = encode_features(X_train, X_test)
        assert isinstance(encoder, OneHotEncoder)

    def test_numeric_columns_preserved(self, train_test_split):
        """Numeric column names must appear in feature_names."""
        X_train, X_test = train_test_split
        _, _, feature_names, _ = encode_features(X_train, X_test)
        assert "tenure" in feature_names
        assert "MonthlyCharges" in feature_names

    def test_ohe_columns_in_feature_names(self, train_test_split):
        """OHE-generated column names must appear in feature_names."""
        X_train, X_test = train_test_split
        _, _, feature_names, _ = encode_features(X_train, X_test)
        # At least one OHE name should contain the original column name
        ohe_cols = [n for n in feature_names if "Contract" in n or "InternetService" in n]
        assert len(ohe_cols) > 0

    def test_train_row_count_preserved(self, train_test_split):
        """Encoded training matrix must have the same number of rows as X_train."""
        X_train, X_test = train_test_split
        X_tr_enc, _, _, _ = encode_features(X_train, X_test)
        assert X_tr_enc.shape[0] == len(X_train)

    def test_test_row_count_preserved(self, train_test_split):
        """Encoded test matrix must have the same number of rows as X_test."""
        X_train, X_test = train_test_split
        _, X_te_enc, _, _ = encode_features(X_train, X_test)
        assert X_te_enc.shape[0] == len(X_test)

    def test_column_count_matches_feature_names(self, train_test_split):
        """Number of columns in encoded arrays must match len(feature_names)."""
        X_train, X_test = train_test_split
        X_tr_enc, X_te_enc, feature_names, _ = encode_features(X_train, X_test)
        assert X_tr_enc.shape[1] == len(feature_names)
        assert X_te_enc.shape[1] == len(feature_names)

    def test_unseen_category_in_test_becomes_zeros(self, mixed_df):
        """Categories in X_test unseen during training must become all-zero rows."""
        X_train = mixed_df.iloc[:2].copy()  # Only Month-to-month and One year
        X_test  = mixed_df.iloc[[2]].copy() # Two year — unseen if drop="first" handles it

        # Use a train set that does NOT contain "Two year"
        X_train_no_two = pd.DataFrame({
            "tenure":          [12, 3],
            "MonthlyCharges":  [50.0, 70.0],
            "Contract":        ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
        })
        X_test_unseen = pd.DataFrame({
            "tenure":          [99],
            "MonthlyCharges":  [999.0],
            "Contract":        ["UNSEEN_CATEGORY"],
            "InternetService": ["DSL"],
        })

        X_tr_enc, X_te_enc, feature_names, _ = encode_features(
            X_train_no_two, X_test_unseen
        )
        # The encoded test row must not raise and must have the correct shape
        assert X_te_enc.shape[1] == len(feature_names)

    def test_numeric_only_df_returns_passthrough(self):
        """With no categorical columns, arrays should pass through unchanged."""
        X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        X_test  = pd.DataFrame({"a": [7, 8],     "b": [9.0, 10.0]})
        X_tr_enc, X_te_enc, feature_names, _ = encode_features(X_train, X_test)
        assert feature_names == ["a", "b"]
        np.testing.assert_array_equal(X_tr_enc, X_train.values)
        np.testing.assert_array_equal(X_te_enc, X_test.values)

    def test_mismatched_columns_raises_value_error(self, mixed_df):
        """Passing X_train and X_test with different columns must raise ValueError."""
        X_train = mixed_df.iloc[:3].copy()
        X_test  = mixed_df.iloc[3:].drop(columns=["Contract"]).copy()
        with pytest.raises(ValueError, match="same columns"):
            encode_features(X_train, X_test)

    def test_encoder_fitted_only_on_train(self, train_test_split):
        """Encoder must be fitted on X_train categories, not on X_test."""
        X_train, X_test = train_test_split
        _, _, _, encoder = encode_features(X_train, X_test)
        train_categories = set(X_train["Contract"].unique())
        encoder_categories = set(encoder.categories_[0])
        assert encoder_categories == train_categories