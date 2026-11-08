"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.5.0
"""
import pandas as pd
import numpy as np


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

def aggregate_pos_cash_balance(pos_cash_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to one row per SK_ID_CURR."""
    agg = pos_cash_balance.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"), # how many times the applicant has a POS_CASH_balance (monthly balance records)
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"), # average number of installments
        POS_SK_DPD_MEAN=("SK_DPD", "mean"), # average number of days past due
        POS_SK_DPD_MAX=("SK_DPD", "max"), # maximum number of days past due
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"), # average number of days past due for the defaulted loans
    ).reset_index()
    return agg

def aggregate_installments_payments(installments_payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments to one row per SK_ID_CURR."""
    installments_payments = installments_payments.copy()
    installments_payments["DAYS_LATE"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
    installments_payments["PAYMENT_DIFF"] = installments_payments["AMT_INSTALMENT"] - installments_payments["AMT_PAYMENT"]

    agg = installments_payments.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT=("SK_ID_PREV", "count"), # how many installment payment recorded per applicant
        INSTAL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"), # average number of days late
        INSTAL_DAYS_LATE_MAX=("DAYS_LATE", "max"), # maximum number of days late
        INSTAL_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"), # average difference between installment and payment amounts
    ).reset_index()
    return agg

def aggregate_credit_card_balance(credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance to one row per SK_ID_CURR."""
    credit_card_balance = credit_card_balance.copy()
    credit_card_balance["UTILIZATION"] = credit_card_balance["AMT_BALANCE"] / credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"]
    # Utilization is the ratio of the balance owed divided by the credit limit (how much of the available credit is being used)
    credit_card_balance["UTILIZATION"] = credit_card_balance["UTILIZATION"].replace([np.inf, -np.inf], np.nan)  
    # replace inf (cases where theres no credit limit) values with NaN

    agg = credit_card_balance.groupby("SK_ID_CURR").agg(
        CC_BALANCE_COUNT=("SK_ID_PREV", "count"), # total months of activity logged
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"), # average balance of the applicant's credit card
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"), # average utilization rate
        CC_SK_DPD_MEAN=("SK_DPD", "mean"), # average number of days past due
        CC_SK_DPD_MAX=("SK_DPD", "max"), # maximum number of days past due
    ).reset_index()
    return agg

def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau_balance to one row per SK_ID_BUREAU."""
    bureau_balance = bureau_balance.copy()
    bureau_balance["IS_DPD"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"])

    agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        BUREAU_BALANCE_COUNT=("MONTHS_BALANCE", "count"), # total months of activity logged
        BUREAU_BALANCE_IS_DPD_COUNT=("IS_DPD", "sum"), # average is_dpd of the applicant's bureau balance
    ).reset_index()
    return agg

def aggregate_bureau(bureau: pd.DataFrame, bureau_balance_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge bureau_balance aggregates onto bureau, then aggregate to one row per SK_ID_BUREAU."""
    bureau = bureau.copy()
    bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"), # total number of bureau records
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"), # average amount of credit
        BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"), # average amount of debt
        BUREAU_CREDIT_DAY_OVERDUE_MEAN =("CREDIT_DAY_OVERDUE", "mean"), # average number of days past due
        BUREAU_BB_DPD_MONTHS_MEAN=("BUREAU_BALANCE_IS_DPD_COUNT", "mean"), # average number of months with days past due
    ).reset_index()
    return agg

# Merging all aggregated tables onto 'application_train through 'SK_ID_CURR'
# using a left join to keep all applicants in the training set, even if they don't have previous applications or credit history

def merge_aggregated_features(
        application: pd.DataFrame,
        previous_application_agg: pd.DataFrame,
        pos_cash_balance_agg: pd.DataFrame,
        installments_payments_agg: pd.DataFrame,
        credit_card_balance_agg: pd.DataFrame,
        bureau_agg: pd.DataFrame
    ) -> pd.DataFrame:
    """Merge all aggregated features onto the application DataFrame."""
    merged = application.merge(previous_application_agg, on="SK_ID_CURR", how="left")
    merged = merged.merge(pos_cash_balance_agg, on="SK_ID_CURR", how="left")
    merged = merged.merge(installments_payments_agg, on="SK_ID_CURR", how="left")
    merged = merged.merge(credit_card_balance_agg, on="SK_ID_CURR", how="left")
    merged = merged.merge(bureau_agg, on="SK_ID_CURR", how="left")
    return merged   
