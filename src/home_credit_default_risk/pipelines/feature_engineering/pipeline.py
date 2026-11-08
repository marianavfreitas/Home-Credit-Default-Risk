"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.5.0
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (aggregate_application, aggregate_pos_cash_balance, 
                    aggregate_installments_payments, aggregate_credit_card_balance, 
                    aggregate_bureau_balance, aggregate_bureau, merge_aggregated_features)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_application,
                inputs="previous_application_processed",
                outputs="previous_application_agg",
                name="aggregate_application_node",
            ),
            node(
                func = aggregate_pos_cash_balance,
                inputs= "POS_CASH_balance_processed",
                outputs = "POS_CASH_balance_agg",
                name = "aggregate_pos_cash_balance_node"
             ),
            node(
                func = aggregate_installments_payments,
                inputs = "installments_payments_processed",
                outputs = "installments_payments_agg",
                name = "aggregate_installments_payments_node"
            ),
            node(
                func = aggregate_credit_card_balance,
                inputs = "credit_card_balance_processed",
                outputs = "credit_card_balance_agg",
                name = "aggregate_credit_card_balance_node"
            ),
            node(
                func = aggregate_bureau_balance,
                inputs = "bureau_balance_processed",
                outputs = "bureau_balance_agg",
                name = "aggregate_bureau_balance_node"
            ),
            node(
                func = aggregate_bureau,
                inputs = ["bureau_processed", "bureau_balance_agg"],
                outputs = "bureau_agg",
                name = "aggregate_bureau_node"
            ),
            node(
                func = merge_aggregated_features,
                inputs = ["application_train_processed", "previous_application_agg", "POS_CASH_balance_agg", 
                          "installments_payments_agg", "credit_card_balance_agg", "bureau_agg"],
                outputs = "merged_features",
                name = "merge_aggregated_features_node"
            )
        ]
    )

