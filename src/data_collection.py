import pandas as pd

from typing import List

def get_input_data(data_path: str = "./data/Telco_customer_churn.csv",
                   columns_to_keep: List[str] = None) -> pd.DataFrame:
    """
    Reads the input data from a CSV file.

    Args:
        data_path (str): The path to the CSV file.
        columns_to_keep (List[str]): A list of column names to keep in the DataFrame.
        If None, all columns will be kept.

    Returns:
        pd.DataFrame: The input data as a pandas DataFrame.
    """
    df = pd.read_csv(data_path)

    if columns_to_keep is not None:
        df = df[columns_to_keep]
        
    return df

def main():
    df_churn_data = get_input_data()

    print(df_churn_data.head())

if __name__ == "__main__":
    main()