import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Union, Iterable, Tuple

import pandas as pd

from datasets.base import Dataset
from pipelines.month import MonthPriceSalesPipeline


logger = logging.getLogger(__name__)


class MonthPriceSalesDataset(Dataset):
    CATEGORICAL_FEATURES = [
        "item_id",
        "shop_id",
        "city_id",
        "country_part",
        "item_category_id",
        "is_new_item",
        "is_first_shop_transaction"
    ]
    VALIDATION_MONTH = 32
    TEST_MONTH = 33

    def __init__(
            self,
            data_dir: str,
            pipeline: MonthPriceSalesPipeline,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / ".cache"

        self.pipeline = pipeline
        self._splits: Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None

    @classmethod
    def from_config(cls, config: dict, data_dir: str, *args, **kwargs) -> "Dataset":
        pipeline_config = config["dataset"]["pipeline"]
        pipeline = MonthPriceSalesPipeline.from_config(pipeline_config)

        return cls(
            data_dir=data_dir,
            pipeline=pipeline
        )

    def load(self):
        data = self.try_load_from_cache()
        if data is None:
            logger.info("Cache not found / invalid, computing dataset from scratch.")

            data = self.read()
            data = self.pipeline.transform(data)
            self.save_to_cache(data)
        else:
            logger.info("Loading data from cache.")

        x, y = self.pick_labels(data, label_columns=["item_sales"])
        splits = self.split(x, y)
        splits = self.scale(splits)

        self._splits = splits

    def get(self, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self._splits is not None, "Dataset not loaded yet. Call load() first."

        return self._splits[split]

    def read(self) -> pd.DataFrame:
        items_df = pd.read_csv(self.data_dir / "items.csv")
        item_categories_df = pd.read_csv(self.data_dir / "item_categories.csv")
        items_df = items_df.merge(item_categories_df, on="item_category_id")
        shops = pd.read_csv(self.data_dir / "shops.csv")

        df = pd.read_csv(self.data_dir / "sales_train.csv")
        df = df.merge(items_df, on="item_id", how="left")
        df = df.merge(shops, on="shop_id", how="left")

        return df

    def try_load_from_cache(self) -> Optional[pd.DataFrame]:
        if not self.cache_dir.exists():
            return None

        meta_file = self.cache_dir / "meta.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r") as f:
            try:
                meta = json.load(f)
            except ValueError:
                return None

        data_file = self.cache_dir / "data.feather"
        if not data_file.exists():
            return None
        df = pd.read_feather(data_file)
        df = df.drop(columns=["index"])

        is_subset = self._is_subset(cached_features=meta["features"])
        has_same_length = len(df) == meta["n_rows"]
        has_same_class_state = set(MonthPriceSalesPipeline.get_class_state()) == set(meta["class_state"])
        if not (is_subset and has_same_length and has_same_class_state):
            shutil.rmtree(self.cache_dir)  # Invalidate cache.

            return None

        return df

    def _is_subset(self, cached_features: List[Dict[str, Union[str, dict]]]) -> bool:
        # TODO: A better check would be relative to the pipeline.__dict__ object
        #  that reflects the whole state of the pipeline.
        current_features = self.pipeline.features

        cached_features = {f["name"]: f for f in cached_features}
        for feature_config in current_features:
            # Check if the feature is in the cached features.
            feature_name = feature_config["name"]
            is_feature_in_cache = feature_name in cached_features
            if not is_feature_in_cache:
                return False

            # Check if the feature parameters are the same.
            feature_parameters = feature_config.get("parameters", dict())
            cached_feature_parameters = cached_features[feature_name].get("parameters", dict())
            has_same_parameters = feature_parameters == cached_feature_parameters
            if not has_same_parameters:
                return False

        return True

    def save_to_cache(self, df: pd.DataFrame) -> None:
        features = self.pipeline.features

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        meta_file = self.cache_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(
                {
                    "n_rows": len(df),
                    "features": features,
                    "class_state": MonthPriceSalesPipeline.get_class_state()
                },
                f
            )

        data_file = self.cache_dir / "data.feather"
        df.reset_index().to_feather(data_file)

    def pick_labels(self, data: pd.DataFrame, label_columns: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x = data.drop(columns=label_columns)
        y = data[label_columns]

        return x, y

    def split(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        train_mask = x["date_block_num"] < min(self.VALIDATION_MONTH, self.TEST_MONTH)
        validation_mask = x["date_block_num"] == self.VALIDATION_MONTH
        test_mask = x["date_block_num"] == self.TEST_MONTH

        # TODO: Find a better way to drop "date_block_num" column in the pipeline process.
        x = x.drop(columns=["date_block_num"])

        x_train = x[train_mask]
        y_train = y[train_mask]

        x_validation = x[validation_mask]
        y_validation = y[validation_mask]

        x_test = x[test_mask]
        y_test = y[test_mask]

        return {
            "train": (x_train, y_train),
            "validation": (x_validation, y_validation),
            "test": (x_test, y_test)
        }

    def scale(self, splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame]]:
        return splits


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
    pipeline = MonthPriceSalesPipeline(features=features)
    dataset = MonthPriceSalesDataset(
        data_dir="../../data/",
        pipeline=pipeline
    )
    data = dataset.load()
