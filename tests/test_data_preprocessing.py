"""tests/test_data_preprocessing.py

Unit tests for src.data_preprocessing.cast_yes_no_variables_to_binary.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import cast_yes_no_variables_to_binary


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame with Yes/No columns."""
    return pd.DataFrame({
        "customerID":      ["0001", "0002", "0003", "0004"],
        "Partner":         ["Yes", "No", "Yes", "No"],
        "Dependents":      ["No", "No", "Yes", "Yes"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No"],
        "Churn":           ["No", "Yes", "No", "Yes"],
        "MonthlyCharges":  [50.0, 70.0, 90.0, 30.0],
    })


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCastYesNoVariablesToBinary:

    def test_returns_dataframe(self, sample_df):
        """Function must return a pandas DataFrame."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Partner"])
        assert isinstance(result, pd.DataFrame)

    def test_yes_maps_to_one(self, sample_df):
        """'Yes' values must be mapped to 1."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Partner"])
        assert result.loc[result.index[0], "Partner"] == 1
        assert result.loc[result.index[2], "Partner"] == 1

    def test_no_maps_to_zero(self, sample_df):
        """'No' values must be mapped to 0."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Partner"])
        assert result.loc[result.index[1], "Partner"] == 0
        assert result.loc[result.index[3], "Partner"] == 0

    def test_multiple_columns_encoded(self, sample_df):
        """All specified columns must be encoded."""
        cols = ["Partner", "Dependents", "PaperlessBilling", "Churn"]
        result = cast_yes_no_variables_to_binary(sample_df.copy(), cols)
        for col in cols:
            assert set(result[col].unique()).issubset({0, 1}), (
                f"Column '{col}' contains values other than 0 and 1."
            )

    def test_non_encoded_columns_unchanged(self, sample_df):
        """Columns not in the list must remain untouched."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Partner"])
        pd.testing.assert_series_equal(
            result["MonthlyCharges"],
            sample_df["MonthlyCharges"],
        )

    def test_output_dtype_is_int(self, sample_df):
        """Encoded columns must have integer dtype."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Churn"])
        assert result["Churn"].dtype == int

    def test_row_count_unchanged(self, sample_df):
        """Encoding must not change the number of rows."""
        result = cast_yes_no_variables_to_binary(sample_df.copy(), ["Partner"])
        assert len(result) == len(sample_df)

    def test_empty_columns_list_is_no_op(self, sample_df):
        """Passing an empty column list must not modify the DataFrame."""
        original = sample_df.copy()
        result = cast_yes_no_variables_to_binary(sample_df.copy(), [])
        pd.testing.assert_frame_equal(result, original)