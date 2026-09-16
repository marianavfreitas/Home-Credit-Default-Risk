import pandas as pd
import numpy as np

def downcast_dataframe (df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast the numeric columns of a DataFrame to reduce memory usage.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be downcasted.

    Returns:
    pd.DataFrame: The downcasted DataFrame with reduced memory usage.
    """
    for col in df.select_dtypes("integer").columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes("object").columns:
        if df[col].nunique() / len(df) < 0.5:  # Only downcast if the number of unique values is less than 50% of the total rows
            df[col] = df[col].astype('category')
    return df

def impute_missing_values (X_train: pd.DataFrame, missing_threshold: float = 0.5):
    """
    Clean meaningless values, drop columns with too many missing values and fill missing values
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Clean meaningless values (e.g., 365243 in DAYS_EMPLOYED)
    X_train["DAYS_EMPLOYED"] = X_train["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_test["DAYS_EMPLOYED"] = X_test["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Drop columns with too many missing values
    kept_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    percentage = X_train.isnull().mean()
    cols_to_drop = [col for col in percentage[percentage > missing_threshold].index if col not in kept_cols]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    # Fill missing values
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].median())

    # Filling categorical missing values with the most common value (mode) of each column
    X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
    X_test[categorical_cols] = X_test[categorical_cols].fillna(X_test[categorical_cols].mode().iloc[0])
    
    return X_train, X_test