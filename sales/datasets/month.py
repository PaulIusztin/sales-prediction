import datetime
import json
import logging
import shutil

from typing import Optional, List, Dict, Union, Iterable, Tuple, Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utils
from datasets.base import Dataset
from pipelines.month import MonthPriceSalesPipeline


logger = logging.getLogger(__name__)


class MonthPriceSalesDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            pipeline: MonthPriceSalesPipeline,
            split_info: Dict[str, Any]
    ):
        super().__init__(pipeline, root_dir, split_info)

        self.cache_dir = self.root_dir / ".cache" / self.pipeline.name

        self.scaler = MinMaxScaler()
        self._unscaled_splits: Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None
        self._scaled_splits: Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None

    def load(self):
        data = self.try_load_from_cache()
        if data is None:
            logger.info("Cache not found / invalid, computing dataset from scratch.")

            data = self.read()
            data = self.pipeline.transform(data)

            logger.info("Caching dataset.")
            self.save_to_cache(data)
        else:
            logger.info("Loading data from cache.")
            data.info()

        x, y = self.pick_labels(data, label_columns=["item_sales"])
        # TODO: Can we find a better way to keep the unscaled splits (instead of a copy)?
        #  Just the inverse_scale() wont work in all the cases
        #  (e.g. it wont inverse 100% the validation & test split we need for some baselines, like the PersistenceModel)
        self._unscaled_splits = self.split(x, y)
        # First create a copy of the splits, then scale them.
        self._scaled_splits = {k: (X.copy(), y.copy()) for k, (X, y) in self._unscaled_splits.items()}
        self._scaled_splits = self.scale(self._scaled_splits)

    def get(self, split: str, scaled: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self._unscaled_splits is not None and self._scaled_splits is not None, \
            "Dataset not loaded yet. Call load() first."

        if scaled is True:
            return self._scaled_splits[split]

        return self._unscaled_splits[split]

    def read(self) -> pd.DataFrame:
        items_df = pd.read_csv(self.root_dir / "items.csv")
        item_categories_df = pd.read_csv(self.root_dir / "item_categories.csv")
        items_df = items_df.merge(item_categories_df, on="item_category_id")
        shops = pd.read_csv(self.root_dir / "shops.csv")

        df = pd.read_csv(self.root_dir / "sales_train.csv")
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

        is_subset = self._is_subset(cached_features=meta["features"])
        has_same_class_name = self.pipeline.__class__.__name__ == meta["class_name"]
        has_same_class_state = set(self.pipeline.get_class_state()) == set(meta["class_state"])
        # has_same_object_state = self.pipeline.get_state() == meta["object_state"]
        if not (is_subset and has_same_class_name and has_same_class_state):
            shutil.rmtree(self.cache_dir)  # Invalidate cache.

            return None

        df = pd.read_feather(data_file)
        df = df.drop(columns=["index"])

        return df

    def _is_subset(self, cached_features: List[Dict[str, Union[str, dict]]]) -> bool:
        current_features = self.pipeline.features

        cached_features = {f["name"]: f for f in cached_features}
        for feature_config in current_features:
            # Check if the feature is in the cached features.
            feature_name = feature_config["name"]
            is_feature_in_cache = feature_name in cached_features
            if not is_feature_in_cache:
                return False

            # Check if the feature parameters are the same.
            feature_parameters = utils.to_consistent_types(feature_config.get("parameters", dict()))
            cached_feature_parameters = utils.to_consistent_types(cached_features[feature_name].get("parameters", dict()))
            has_same_parameters = feature_parameters == cached_feature_parameters
            if not has_same_parameters:
                return False

        return True

    def save_to_cache(self, df: pd.DataFrame) -> None:
        features = self.pipeline.features

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TODO: A better check would be relative to the pipeline.__dict__ object
        #  that reflects the whole internal state of the pipeline, otherwise changes to a function wont be reflected.
        meta_file = self.cache_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(
                {
                    "n_rows": len(df),
                    "features": features,
                    "class_name": self.pipeline.__class__.__name__,
                    "class_state": self.pipeline.get_class_state(),
                    # "object_state": self.pipeline.get_state(),
                    "datetime": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                },
                f
            )
        csv_data_file = self.cache_dir / "data.csv"
        df.to_csv(csv_data_file, index=False)

        feather_data_file = self.cache_dir / "data.feather"
        df.reset_index().to_feather(feather_data_file)

    def pick_labels(self, data: pd.DataFrame, label_columns: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x = data.drop(columns=label_columns)
        y = data[label_columns]

        return x, y

    def split(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        train_mask = x["date_block_num"] < min(self.split_info["validation"], self.split_info["test"])
        validation_mask = x["date_block_num"] == self.split_info["validation"]
        test_mask = x["date_block_num"] == self.split_info["test"]

        # TODO: Find a better way to drop "date_block_num" column in the pipeline process.
        x = x.drop(columns=["date_block_num"])

        x_train = x[train_mask]
        y_train = y[train_mask]

        x_validation = x[validation_mask]
        y_validation = y[validation_mask]

        x_test = x[test_mask]
        y_test = y[test_mask]

        assert all([len(x_train) != 0, len(x_validation) != 0, len(x_test) != 0]), "All splits must have data."

        return {
            "train": (x_train, y_train),
            "validation": (x_validation, y_validation),
            "test": (x_test, y_test)
        }

    def scale(
            self,
            splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        train_x, train_y = splits["train"]
        validation_x, validation_y = splits["validation"]
        test_x, test_y = splits["test"]

        self.scaler.fit(train_x)

        scaled_train_values = self.scaler.transform(train_x)
        scaled_validation_values = self.scaler.transform(validation_x)
        scales_test_values = self.scaler.transform(test_x)

        columns = train_x.columns
        splits["train"] = (pd.DataFrame(scaled_train_values, columns=columns), train_y)
        splits["validation"] = (pd.DataFrame(scaled_validation_values, columns=columns), validation_y)
        splits["test"] = (pd.DataFrame(scales_test_values, columns=columns), test_y)

        return splits

    def inverse_scale(
            self, splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        inverted_splits = dict()
        for split_name, split_value in splits.items():
            x, y = split_value

            columns = x.columns
            inverted_x = self.scaler.inverse_transform(x)
            inverted_x = pd.DataFrame(inverted_x, columns=columns)

            inverted_splits[split_name] = (inverted_x, y)

        return inverted_splits
