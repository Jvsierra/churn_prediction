import pandas as pd

def get_input_data(data_path: str = "./data/Telco_customer_churn.csv") -> pd.DataFrame:
    """
    Reads the input data from a CSV file.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The input data as a pandas DataFrame.
    """
    return pd.read_csv(data_path)

def main():
    df_churn_data = get_input_data()

    print(df_churn_data.head())

if __name__ == "__main__":
    main()