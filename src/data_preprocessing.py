import pandas as pd

from typing import List

def cast_yes_no_variables_to_binary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Casts yes/no variables to binary (1/0) in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the columns to be cast.
    columns (List[str]): A list of column names that contain yes/no values.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns cast to binary.
    """
    for column in columns:
        df[column] = df[column].map({'yes': 1, 'no': 0})

    return df