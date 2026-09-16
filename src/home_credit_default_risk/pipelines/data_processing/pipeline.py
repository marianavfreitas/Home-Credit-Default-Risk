from kedro.pipeline import Pipeline, node, pipeline 

from .nodes import downcast_dataframe

DATASETS = [
    "application_train",
    "application_test",
    "previous_application",
    "bureau",
    "bureau_balance",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=downcast_dataframe,
                inputs=f"{name}_raw",
                outputs=f"{name}_processed",
                name=f"downcast_{name}_node",
            )
            for name in DATASETS
        ]
    )