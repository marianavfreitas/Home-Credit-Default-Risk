"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.5.0
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    clip_outliers, evaluate_model, split_data, impute_missing_values, 
    scale_features, encode_categorical_features, train_model, select_features,
    train_logistic_regression, train_random_forest, train_random_forest_tuned, train_xgboost
    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = split_data,
                inputs = ["merged_features", "params:test_size", "params:random_state"],
                outputs = ["X_train", "X_test", "y_train", "y_test"],
                name = "split_data_node",
            ),
            node(
                func = impute_missing_values,
                inputs = ["X_train", "X_test"],
                outputs = ["X_train_imputed", "X_test_imputed"],
                name = "impute_missing_values_node",
            ),
            node(
                func = clip_outliers,
                inputs = ["X_train_imputed", "X_test_imputed"],
                outputs = ["X_train_capped", "X_test_capped"],
                name = "clip_outliers_node",
            ),
            node(
                func = scale_features,
                inputs = ["X_train_capped", "X_test_capped"],
                outputs = ["X_train_scaled", "X_test_scaled"],
                name = "scale_features_node",
            ),
            node(
                func = encode_categorical_features,
                inputs = ["X_train_scaled", "X_test_scaled"],
                outputs = ["X_train_encoded", "X_test_encoded"],
                name = "encode_categorical_features_node",
            ),
            node(
                func = train_model,
                inputs = ["X_train_encoded", "y_train"],
                outputs = "model",
                name = "train_model_node",
            ),
            node(
                func = evaluate_model,
                inputs = ["model", "X_test_encoded", "y_test"],
                outputs = "model_metrics",
                name = "evaluate_model_node",
            ),
            node(
                func = select_features,
                inputs = ["X_train_encoded", "X_test_encoded", "model"],
                outputs = ["X_train_selected", "X_test_selected"],
                name = "select_features_node",
            ),
            node(
                func = train_model,
                inputs = ["X_train_selected", "y_train"],
                outputs = "model_selected_features",
                name = "train_model_selected_features_node",
            ),
            node(
                func = evaluate_model,
                inputs = ["model_selected_features", "X_test_selected", "y_test"],
                outputs = "model_metrics_selected",
                name = "evaluate_model_selected_node",
            ),
            node(
                func = train_logistic_regression,
                inputs = ["X_train_encoded", "y_train"],
                outputs = "model_logreg",
                name = "train_logistic_regression_node",
            ),
            node(
                func = train_random_forest,
                inputs = ["X_train_encoded", "y_train"],
                outputs = "model_rf",
                name = "train_random_forest_node",
            ),
            node(
                func = train_random_forest_tuned,
                inputs = ["X_train_encoded", "y_train"],
                outputs = "model_rf_tuned",
                name = "train_random_forest_tuned_node",
            ),
            node(
                func = train_xgboost,
                inputs = ["X_train_encoded", "y_train"],
                outputs = "model_xgb",
                name = "train_xgboost_node",
            )
        ]
    )

