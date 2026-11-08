import re

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb


from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Data Split

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

"------------------------------------------------------------------------------------------------------------------------------"

# Missing Values Imputation

def impute_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame, missing_threshold: float = 0.5):
    """Clean sentinel values, drop mostly-empty columns, fill the rest — fitting only on X_train."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["DAYS_EMPLOYED"] = X_train["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_test["DAYS_EMPLOYED"] = X_test["DAYS_EMPLOYED"].replace(365243, np.nan)

    X_train["CC_HAS_CARD"] = X_train["CC_BALANCE_COUNT"].notna().astype(int)
    X_test["CC_HAS_CARD"] = X_test["CC_BALANCE_COUNT"].notna().astype(int)

    columns_to_keep = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", 
        "CC_BALANCE_COUNT", "CC_AMT_BALANCE_MEAN", "CC_UTILIZATION_MEAN",
        "CC_SK_DPD_MEAN", "CC_SK_DPD_MAX"
        ]
    
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

"------------------------------------------------------------------------------------------------------------------------------"

# Outlier Treatment

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

"------------------------------------------------------------------------------------------------------------------------------"

# Scaling Numerical Features

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale features using StandardScaler, fitting only on X_train."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test

"------------------------------------------------------------------------------------------------------------------------------"

# Encoding Categorical Features

def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Encode categorical features using OneHotEncoder, fitting only on X_train."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output = False)
    encoder.fit(X_train[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)
    feature_names = [re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in feature_names]

    train_encoded = pd.DataFrame(
        encoder.transform(X_train[categorical_cols]),
        columns=feature_names,
        index=X_train.index
    )
    test_encoded = pd.DataFrame(
        encoder.transform(X_test[categorical_cols]),
        columns=feature_names,
        index=X_test.index
    )   
    # Drop original categorical columns and concatenate the encoded ones
    X_train = X_train.drop(columns=categorical_cols).join(train_encoded)
    X_test = X_test.drop(columns=categorical_cols).join(test_encoded)

    return X_train, X_test

# Due to the encoding of the categorical features, the number of columns of X_train and X_test
# may increase, because each categorical feature is transformed into multiple binary features (one for each category)
"------------------------------------------------------------------------------------------------------------------------------"

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train a baseline LightGBM classifier."""
    model = lgb.LGBMClassifier(
        random_state=42,
        objective = "binary",
        is_unbalance = True, 
        num_leaves = 15,            # less trees = simple trees = less overfitting
        min_child_samples = 50,     # each leaf needs more data to form  = less noise
        reg_alpha = 0.1,            # L1 regularization
        reg_lambda = 0.1,           # L2 regularization
        subsample = 0.8,            # each tree sees only 80% of the data = less overfitting
        colsample_bytree = 0.8,     # each tree sees only 80% of the features = less overfitting
        )
    model.fit(X_train, y_train["TARGET"])
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """Evaluate the model using ROC AUC score."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test["TARGET"], y_pred_proba)
    print(f"Test ROC AUC Score: {auc:.4f}")

    return {"auc_roc": auc}

# We will use auc roc as the main metric for model evaluation, as it is a common metric 
# for binary classification problems, especially in imbalanced datasets. 
# It measures the ability of the model to distinguish between the positive and negative 
# classes across different threshold values.

"------------------------------------------------------------------------------------------------------------------------------"

def select_features(X_train: pd.DataFrame, X_test: pd.DataFrame, model):
    """Keeps only features the baseline model used (importance > 0)"""
    importances = pd.Series(model.feature_importances_, index=model.feature_name_)
    selected_features = importances[importances > 0].index.tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected

"------------------------------------------------------------------------------------------------------------------------------"

def evaluate_classification(model, x, y, split = ""):
    """Evaluate the model using various classification metrics."""
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:, 1]

    return {
        "split": split,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_pred_proba)
    }

"------------------------------------------------------------------------------------------------------------------------------"

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train a Logistic Regression baseline."""
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        n_jobs=-1
    )
    model.fit(X_train, y_train["TARGET"])
    return model


"------------------------------------------------------------------------------------------------------------------------------"

def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train a Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators= 300,
        max_depth= 10,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs = -1,
    )
    model.fit(X_train, y_train["TARGET"])
    return model

def train_random_forest_tuned(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train a Random Forest with hyperparameter tuning using RandomizedSearchCV."""
    
    model = RandomForestClassifier(
            n_estimators= 300,
            max_depth= 10,
            min_samples_leaf=100,
            min_samples_split= 20,
            max_features= 'sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs = -1,
        )
    model.fit(X_train, y_train["TARGET"])
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train an XGBoost classifier."""
    scale_pos_weight = (y_train["TARGET"] == 0).sum() / (y_train["TARGET"] == 1).sum()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        min_child_weight=10,     # similar role to min_samples_leaf — needs more data per leaf
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=0.1,          # L2 regularization
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train["TARGET"])
    return model
