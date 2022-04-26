from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import utils


class MonthPriceSalesPipeline:
    DROP_COLUMNS = [
        "item_name",
        "item_category_name",
        "shop_name",
        "city_name",
        "date",
    ]

    def __init__(
            self,
            features: List[Dict[str, Union[str, dict]]],
            drop_columns: bool = True,
            drop_rows: bool = True
    ):
        self.features = features
        self.dict_features = {f["name"]: f for f in features}  # Keep a dict version of the features for easier lookup.
        self.drop_columns = drop_columns
        self.drop_rows = drop_rows

        self.supported_features = {
            "city": self._add_city_feature,
            "is_new_item": self._add_is_new_item_feature,
            "is_first_shop_transaction": self._add_is_first_shop_transaction_feature,
            "category_sales": self._add_category_sales_feature,
            "lags": self._add_multiple_lag_feature
        }
        assert all([feature["name"] in self.supported_features for feature in features]), "Features not supported."

    @classmethod
    def from_config(cls, config: dict) -> "MonthPriceSalesPipeline":
        parameters = config["parameters"]

        return cls(**parameters)

    @classmethod
    def get_class_state(cls) -> list:
        return list(MonthPriceSalesPipeline.__dict__.keys())

    def get_state(self) -> list:
        return list(self.__dict__.keys())

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.clean(data)
        data = self.aggregate(data)
        data = self.add_features(data)
        data = self.drop(data)

        return data

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        assert data.isna().sum().sum() == 0, "We expect no missing values."

        data = self._cast_columns(data)
        data = utils.remove_outliers_threshold(data, {
            "item_cnt_day": {
                "min": 0,
                "max": 1000,
            }
        })
        data = utils.remove_outliers_iqr(data, ["item_price"])

        return data

    @classmethod
    def _cast_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        data["date"] = pd.to_datetime(data["date"], format="%d.%m.%Y")

        data["date_block_num"] = data["date_block_num"].astype(np.int8)
        data["shop_id"] = data["shop_id"].astype(np.int8)
        data["shop_name"] = data["shop_name"].astype(pd.StringDtype())
        data["item_id"] = data["item_id"].astype(np.int16)
        data["item_name"] = data["item_name"].astype(pd.StringDtype())
        data["item_category_id"] = data["item_category_id"].astype(np.int8)
        data["item_category_name"] = data["item_category_name"].astype(pd.StringDtype())
        data["item_price"] = data["item_price"].astype(np.float32)
        data["item_cnt_day"] = data["item_cnt_day"].astype(np.int16)

        return data

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg(
            date=("date", "min"),
            shop_name=("shop_name", "first"),
            item_category_id=("item_category_id", "first"),
            item_category_name=("item_category_name", "first"),
            item_name=("item_name", "first"),
            item_price=("item_price", "mean"),
            item_sales=("item_cnt_day", "sum"),
            # TODO: Search item_cnt_day_avg in https://www.kaggle.com/code/deinforcement/top-1-predict-future-sales-features-lightgbm
            # item_daily_sales=("item_cnt_day", "mean")
        )

        return data

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for feature_config in (pbar := tqdm(self.features)):
            feature_name = feature_config["name"]
            feature_parameters = feature_config.get("parameters", {})
            pbar.set_description(f"Adding feature '{feature_name}'")

            f = self.supported_features[feature_name]
            data = f(data, **feature_parameters)

        return data

    def drop(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.drop_rows:
            lags = self.dict_features.get("lags", {}).get("parameters", {}).get("lags", [])
            # Drop rows that do not have all the required lags.
            if len(lags) > 0:
                max_lag = max(lags)
                data = data[data["date_block_num"] >= max_lag]

        if self.drop_columns:
            data = data.drop(self.DROP_COLUMNS, axis=1)

        return data

    @classmethod
    def _add_is_new_item_feature(cls, data: pd.DataFrame) -> pd.DataFrame:
        is_new_item_df = data.groupby(["item_id"], as_index=False)["date_block_num"].min()
        is_new_item_df["is_new_item"] = 1

        data = data.merge(
            is_new_item_df[["date_block_num", "item_id", "is_new_item"]],
            on=["date_block_num", "item_id"],
            how="left"
        )
        data["is_new_item"] = data["is_new_item"].fillna(0)
        data["is_new_item"] = data["is_new_item"].astype(np.int8)

        return data

    @classmethod
    def _add_is_first_shop_transaction_feature(cls, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: Add shop_item_sold_before instead?

        is_first_shop_transaction_df = data.groupby(["shop_id", "item_id"], as_index=False)["date_block_num"].min()
        is_first_shop_transaction_df["is_first_shop_transaction"] = 1

        data = data.merge(
            is_first_shop_transaction_df[["date_block_num", "shop_id", "item_id", "is_first_shop_transaction"]],
            on=["date_block_num", "shop_id", "item_id"],
            how="left"
        )
        data["is_first_shop_transaction"] = data["is_first_shop_transaction"].fillna(0)
        data["is_first_shop_transaction"] = data["is_first_shop_transaction"].astype(np.int8)

        return data

    @classmethod
    def _add_city_feature(cls, data: pd.DataFrame) -> pd.DataFrame:
        data["city_name"] = data["shop_name"].apply(lambda x: x.split()[0].lower())
        data.loc[data.city_name == "!якутск", "city_name"] = "якутск"
        data["city_id"] = LabelEncoder().fit_transform(data["city_name"])

        coords = dict()
        coords["якутск"] = (62.028098, 129.732555, 4)
        coords["адыгея"] = (44.609764, 40.100516, 3)
        coords["балашиха"] = (55.8094500, 37.9580600, 1)
        coords["волжский"] = (53.4305800, 50.1190000, 3)
        coords["вологда"] = (59.2239000, 39.8839800, 2)
        coords["воронеж"] = (51.6720400, 39.1843000, 3)
        coords["выездная"] = (0, 0, 0)
        coords["жуковский"] = (55.5952800, 38.1202800, 1)
        coords["интернет-магазин"] = (0, 0, 0)
        coords["казань"] = (55.7887400, 49.1221400, 4)
        coords["калуга"] = (54.5293000, 36.2754200, 4)
        coords["коломна"] = (55.0794400, 38.7783300, 4)
        coords["красноярск"] = (56.0183900, 92.8671700, 4)
        coords["курск"] = (51.7373300, 36.1873500, 3)
        coords["москва"] = (55.7522200, 37.6155600, 1)
        coords["мытищи"] = (55.9116300, 37.7307600, 1)
        coords["н.новгород"] = (56.3286700, 44.0020500, 4)
        coords["новосибирск"] = (55.0415000, 82.9346000, 4)
        coords["омск"] = (54.9924400, 73.3685900, 4)
        coords["ростовнадону"] = (47.2313500, 39.7232800, 3)
        coords["спб"] = (59.9386300, 30.3141300, 2)
        coords["самара"] = (53.2000700, 50.1500000, 4)
        coords["сергиев"] = (56.3000000, 38.1333300, 4)
        coords["сургут"] = (61.2500000, 73.4166700, 4)
        coords["томск"] = (56.4977100, 84.9743700, 4)
        coords["тюмень"] = (57.1522200, 65.5272200, 4)
        coords["уфа"] = (54.7430600, 55.9677900, 4)
        coords["химки"] = (55.8970400, 37.4296900, 1)
        coords["цифровой"] = (0, 0, 0)
        coords["чехов"] = (55.1477000, 37.4772800, 4)
        coords["ярославль"] = (57.6298700, 39.8736800, 2)

        data["city_coord_1"] = data["city_name"].apply(lambda x: coords[x][0])
        data["city_coord_2"] = data["city_name"].apply(lambda x: coords[x][1])
        data["country_part"] = data["city_name"].apply(lambda x: coords[x][2])

        data["city_id"] = data["city_id"].astype(np.int8)
        data["city_name"] = data["city_name"].astype(pd.StringDtype())
        data["city_coord_1"] = data["city_coord_1"].astype(np.float16)
        data["city_coord_2"] = data["city_coord_2"].astype(np.float16)
        data["country_part"] = data["country_part"].astype(np.int8)

        return data

    @classmethod
    def _add_category_sales_feature(cls, data: pd.DataFrame, levels: List[str] = ("company", "city", "shop")) -> pd.DataFrame:
        # TODO: Should we add this feature only to the new items (is_new_item == 1) ?
        for level in levels:
            assert level in ("company", "city", "shop")

            if level == "company":
                level_category_df = data.groupby(["date_block_num", "item_category_id"], as_index=False).agg(
                    category_company_average_item_sales=("item_sales", "mean"),
                    category_company_average_item_price=("item_price", "mean")
                )
                data = data.merge(level_category_df, on=["date_block_num", "item_category_id"], how="left")
            elif level == "city":
                level_category_df = data.groupby(["date_block_num", "city_id", "item_category_id"], as_index=False).agg(
                    category_city_average_item_sales=("item_sales", "mean"),
                    category_city_average_item_price=("item_price", "mean")
                )
                data = data.merge(level_category_df, on=["date_block_num", "city_id", "item_category_id"], how="left")
            else:
                level_category_df = data.groupby(["date_block_num", "shop_id", "item_category_id"], as_index=False).agg(
                    category_shop_average_item_sales=("item_sales", "mean"),
                    category_shop_average_item_price=("item_price", "mean")
                )
                data = data.merge(level_category_df, on=["date_block_num", "shop_id", "item_category_id"], how="left")

        return data

    @classmethod
    def _add_multiple_lag_feature(
            cls,
            data: pd.DataFrame,
            columns: List[str],
            lags: List[int],
            fill_value: int = 0
    ) -> pd.DataFrame:
        for column in columns:
            data = cls._add_lag_feature(data, column, lags, fill_value)

        return data

    @classmethod
    def _add_lag_feature(cls, data: pd.DataFrame, column: str, lags: List[int], fill_value: int = 0) -> pd.DataFrame:
        initial_dtype = data[column].dtype

        for lag in lags:
            lagged_df = data[["date_block_num", "shop_id", "item_id", column]].copy()
            lagged_df.columns = ["date_block_num", "shop_id", "item_id", f"{column}_lag_{lag}"]
            lagged_df["date_block_num"] += lag

            data = data.merge(lagged_df, on=["date_block_num", "shop_id", "item_id"], how="left")
            data = data.fillna(fill_value)
            data[f"{column}_lag_{lag}"] = data[f"{column}_lag_{lag}"].astype(initial_dtype)

        return data
