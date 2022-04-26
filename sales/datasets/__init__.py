from pipelines import MonthPriceSalesPipeline

from .base import Dataset
from .month import MonthPriceSalesDataset


def build(data_directory: str, dataset_config: dict) -> Dataset:
    features = dataset_config["pipeline"]["features"]
    pipeline = MonthPriceSalesPipeline(features=features)

    dataset = MonthPriceSalesDataset(
        data_dir=data_directory,
        pipeline=pipeline
    )

    return dataset
