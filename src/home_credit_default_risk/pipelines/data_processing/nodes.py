import pandas as pd
import numpy as np

def downcast_dataframe (df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast the numeric columns of a DataFrame to reduce memory usage.

    Parameters:
    Input DataFrame to be downcasted.

    Returns:
    The downcasted DataFrame with reduced memory usage.
    """
    for col in df.select_dtypes("integer").columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes("object").columns:
        if df[col].nunique() / len(df) < 0.5:  # Only downcast if the number of unique values is less than 50% of the total rows
            df[col] = df[col].astype('category')
    return df