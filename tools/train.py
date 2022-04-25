from datasets.month import MonthPriceSalesDataset
from pipelines.month import MonthPriceSalesPipeline
from testers.month import MonthSalesTester
from trainers.lightgbm import LightGBMTrainer
from trainers.persistence import PersistenceModelTrainer

if __name__ == "__main__":
    # TODO: How to add revenue & time features?
    features = [
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
            "name": "category_sales"
        },
        {
            "name": "lags",
            "parameters": {
                "columns": ["item_sales", "category_company_average_item_sales"],
                "lags": [1, 2, 3],
                "fill_value": 0
            }
        }
    ]
    output_folder = "../outputs"

    pipeline = MonthPriceSalesPipeline(features=features)
    dataset = MonthPriceSalesDataset(
        data_dir="../data/",
        pipeline=pipeline
    )

    for trainer_class in [PersistenceModelTrainer]:
        print(f"Training {trainer_class.__name__}")
        trainer = trainer_class(output_folder=output_folder, dataset=dataset)
        trainer.train()
        tester = MonthSalesTester(trainer)
        tester.test()
