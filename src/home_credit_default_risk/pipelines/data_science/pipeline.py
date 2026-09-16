"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.5.0
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clip_outliers, split_data, impute_missing_values

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = split_data,
                inputs = ["application_train_processed", "params:test_size", "params:random_state"],
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

])

