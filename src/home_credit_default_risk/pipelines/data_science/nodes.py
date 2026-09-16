import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def split_data(application_train_processed: pd.DataFrame, test_size: float, random_state: int):
    """Split the data into train and test sets.

    Args:
        application_train_processed: Input dataframe to be split.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.
    Returns:
        X_train: Training features.
        X_test: Testing features.  
        y_train: Training target.
        y_test: Testing target.
    """
    X = application_train_processed.drop(columns = ["TARGET", "SK_ID_CURR"])
    # Removed SK_ID_CURR as it is not a feature for training, just an identifier
    y = application_train_processed[["TARGET"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test


def impute_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame, missing_threshold: float = 0.5):
    """Clean sentinel values, drop mostly-empty columns, fill the rest — fitting only on X_train."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["DAYS_EMPLOYED"] = X_train["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_test["DAYS_EMPLOYED"] = X_test["DAYS_EMPLOYED"].replace(365243, np.nan)

    columns_to_keep = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    missing_pct = X_train.isnull().mean()
    cols_to_drop = [col for col in missing_pct[missing_pct > missing_threshold].index if col not in columns_to_keep]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    numeric_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    medians = X_train[numeric_cols].median()
    modes = X_train[categorical_cols].mode().iloc[0]

    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    X_train[categorical_cols] = X_train[categorical_cols].fillna(modes)
    X_test[categorical_cols] = X_test[categorical_cols].fillna(modes)

    return X_train, X_test

def clip_outliers(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Clip extreme values to IQR bounds, computing bounds only from X_train."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns
    cols_to_cap = [col for col in numeric_cols if X_train[col].nunique() > 2] # skips binary columns

    for col in cols_to_cap:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    return X_train, X_test
