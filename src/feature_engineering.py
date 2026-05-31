import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drop: str = "first",
) -> tuple[np.ndarray, np.ndarray, list[str], OneHotEncoder]:
    """Applies one-hot encoding to categorical features.

    Fits the encoder exclusively on X_train to prevent data leakage, then
    transforms both X_train and X_test. Numeric columns are passed through
    unchanged. The encoded arrays are returned as numpy arrays ready for
    model ingestion.

    Args:
        X_train: Training feature matrix. May contain numeric and categorical
            (object / category dtype) columns.
        X_test: Test / validation feature matrix with the same columns as
            X_train. Categories unseen during training become all-zero rows
            (handle_unknown="ignore").
        drop: Strategy for dropping redundant dummy columns to avoid
            multicollinearity. Passed directly to OneHotEncoder. Defaults to
            "first" (drops the first category of each feature).

    Returns:
        A tuple of four elements:
            - X_train_encoded (np.ndarray): Encoded training matrix.
            - X_test_encoded (np.ndarray): Encoded test matrix.
            - feature_names (list[str]): Column names of the encoded matrix,
              combining numeric column names and OHE-generated names.
            - encoder (OneHotEncoder): Fitted encoder instance. Store it to
              transform future inference batches consistently.

    Raises:
        ValueError: If X_train and X_test do not share the same column names.

    Example:
        >>> X_tr_enc, X_te_enc, names, enc = encode_features(X_train, X_test)
        >>> print(names[:5])
    """
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError(
            "X_train and X_test must have the same columns. "
            f"Difference: {set(X_train.columns) ^ set(X_test.columns)}"
        )

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_cols = X_train.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    if categorical_cols:
        encoder = OneHotEncoder(
            drop=drop,
            sparse_output=False,
            handle_unknown="ignore",   # unseen categories become all-zero rows
        )
        # Fit only on training data — never on test data
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat  = encoder.transform(X_test[categorical_cols])

        ohe_names = encoder.get_feature_names_out(categorical_cols).tolist()

        X_train_encoded = np.hstack([X_train[numeric_cols].values, X_train_cat])
        X_test_encoded  = np.hstack([X_test[numeric_cols].values,  X_test_cat])
        feature_names   = numeric_cols + ohe_names
    else:
        # No categorical columns — return numeric arrays unchanged
        encoder         = OneHotEncoder()  # unfitted placeholder for type consistency
        X_train_encoded = X_train[numeric_cols].values
        X_test_encoded  = X_test[numeric_cols].values
        feature_names   = numeric_cols

    return X_train_encoded, X_test_encoded, feature_names, encoder

def main():
    pass

if __name__ == "__main__":
    main()