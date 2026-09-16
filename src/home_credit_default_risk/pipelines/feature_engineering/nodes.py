"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.5.0
"""
import pandas as pd


def aggregate_application(previous_application: pd.DataFrame) -> pd.DataFrame:
    """Aggregate applications to one row per SK_ID_CURR."""
    agg = previous_application.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"), # how many times the applicant applied for credit
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"), # average amount of credit the applicat asks for
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"), # maximum amount of credit the applicant asks for
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"), # average annuity of the applicant' loan
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"), # average number of days since the decision was made
        # this gives us the recency of the applicant's previous applications
    ).reset_index()
    return agg
