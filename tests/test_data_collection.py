import os
import pytest
import pandas as pd

from src.data_collection import get_input_data


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path) -> str:
    """Creates a minimal CSV file that mimics the Telco churn dataset."""
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "0001-AAA,Male,0,Yes,No,12,50.0,600.0,No\n"
        "0002-BBB,Female,1,No,Yes,3,70.0,210.0,Yes\n"
        "0003-CCC,Male,0,Yes,Yes,24,90.0,2160.0,No\n"
    )
    csv_file = tmp_path / "test_churn.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGetInputData:

    def test_returns_dataframe(self, sample_csv):
        """get_input_data must return a pandas DataFrame."""
        df = get_input_data(data_path=sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_all_rows_loaded(self, sample_csv):
        """All rows in the CSV should be loaded when no filter is applied."""
        df = get_input_data(data_path=sample_csv)
        assert len(df) == 3

    def test_all_columns_loaded_when_none(self, sample_csv):
        """All columns should be returned when columns_to_keep is None."""
        df = get_input_data(data_path=sample_csv)
        expected_cols = {
            "customerID", "gender", "SeniorCitizen", "Partner",
            "Dependents", "tenure", "MonthlyCharges", "TotalCharges", "Churn",
        }
        assert set(df.columns) == expected_cols

    def test_columns_to_keep_filters_correctly(self, sample_csv):
        """Only the specified columns should be present in the output."""
        cols = ["customerID", "Churn"]
        df = get_input_data(data_path=sample_csv, columns_to_keep=cols)
        assert list(df.columns) == cols

    def test_columns_to_keep_does_not_drop_rows(self, sample_csv):
        """Filtering columns must not affect the row count."""
        df = get_input_data(data_path=sample_csv, columns_to_keep=["customerID"])
        assert len(df) == 3

    def test_file_not_found_raises(self, tmp_path):
        """A non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_input_data(data_path=str(tmp_path / "missing.csv"))

    def test_values_are_preserved(self, sample_csv):
        """Values read from CSV should match the original data."""
        df = get_input_data(data_path=sample_csv)
        assert df.iloc[0]["customerID"] == "0001-AAA"
        assert df.iloc[1]["Churn"] == "Yes"
        assert df.iloc[2]["tenure"] == 24