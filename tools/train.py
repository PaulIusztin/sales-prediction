import argparse
import logging

from datasets.month import MonthPriceSalesDataset
from pipelines.month import MonthPriceSalesPipeline
from runner import Runner

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='Path to the root directory of the dataset.')
parser.add_argument('--output-dir', required=True, help='Path to the root directory of the output.')

config = {
    "dataset": {
        "name": "MonthPriceSalesDataset",
        "parameters": {},
        "pipeline": {
            "name": "MonthPriceSalesPipeline",
            "parameters": {
                "features": [
                    {
                        "name": "time",
                    },
                    {
                        "name": "revenue",
                    },
                    {
                        "name": "city"
                    },
                    {
                        "name": "is_new_item"
                    },
                    {
                        "name": "is_first_shop_transaction"
                    },
                    {
                        "name": "category_sales",
                        "parameters": {
                            "levels": ("company", "city", "shop")
                        }
                    },
                    {
                        "name": "lags",
                        "parameters": {
                            "item_sales": {
                                "lags": [1, 2, 3],
                                "fill_value": 0
                            },
                            "category_company_average_item_sales": {
                                "lags": [1, 2, 3],
                                "fill_value": 0
                            },
                            "category_company_average_item_price": {
                                "lags": [1, 2],
                                "fill_value": 0
                            },
                            "category_city_average_item_sales": {
                                "lags": [1, 2, 3],
                                "fill_value": 0
                            },
                            "category_city_average_item_price": {
                                "lags": [1],
                                "fill_value": 0
                            },
                            "category_shop_average_item_sales": {
                                "lags": [1, 2, 3],
                                "fill_value": 0
                            },
                            "category_shop_average_item_price": {
                                "lags": [1, 2],
                                "fill_value": 0
                            },
                            "item_revenue": {
                                "lags": [1],
                                "fill_value": 0
                            }
                        }
                    }
                ]
            }
        }
    },
    "models": [
        {
            "name": "persistence",
            "parameters": {
                "predict_column": "item_sales_lag_1"
            }
        },
        {
            "name": "lightgbm",
            "parameters": {}
        },
        {
            "name": "linear_regression",
            "parameters": {}
        },
        {
            "name": "xgboost",
            "parameters": {}
        }
    ]
}


def train(
        config: dict,
        data_dir: str,
        output_dir: str
):
    dataset = MonthPriceSalesDataset.from_config(
        config=config,
        data_dir=data_dir
    )
    runner = Runner.from_config(
        config=config,
        output_dir=output_dir
    )
    runner.run(dataset)


if __name__ == "__main__":
    args = parser.parse_args()

    # TODO: Add a better loging configuration.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    train(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
