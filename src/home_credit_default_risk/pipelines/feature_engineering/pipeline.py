"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.5.0
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import aggregate_application

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_application,
                inputs="previous_application_processed",
                outputs="previous_application_agg",
                name="aggregate_application_node",
            ),
        ]
    )

