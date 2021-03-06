import datetime
import logging
from functools import partial, lru_cache
from itertools import product
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import utils
from pipelines.base import Pipeline

logger = logging.getLogger(__name__)


class MonthPriceSalesPipeline(Pipeline):
    def __init__(
            self,
            features: List[Dict[str, Union[str, dict]]],
            categorical_features: List[str] = None,
            drop_columns: dict = None,
            drop_rows: bool = True
    ):
        self.features = features
        self.dict_features = {f["name"]: f for f in features}  # Keep a dict version of the features for easier lookup.
        self.categorical_features = categorical_features
        self.drop_columns = drop_columns
        self.drop_rows = drop_rows

        self.before_aggregate_supported_features = {
            "revenue": self._add_revenue_feature,
        }
        self.after_aggregate_supported_features = {
            "time": self._add_time_feature,
            "city": self._add_city_feature,
            "is_new_item": self._add_is_new_item_feature,
            "is_first_shop_transaction": self._add_is_first_shop_transaction_feature,
            "category_sales": self._add_category_sales_feature,
            "lags": self._add_multiple_lag_feature
        }
        self.supported_features = {**self.before_aggregate_supported_features, **self.after_aggregate_supported_features}
        assert all(
            [feature["name"] in self.supported_features for feature in features]
        ), "Features not supported."

    @classmethod
    def from_config(cls, config: dict) -> "MonthPriceSalesPipeline":
        parameters = config["parameters"]

        return cls(**parameters)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("The data before the transformations looks like:")
        logger.info(data.info())

        logger.info("Cleaning...")
        data = self.clean(data)
        logger.info("Adding features before aggregation...")
        data = self.add_features(
            data,
            features=[
                feature for feature in self.features if feature["name"] in self.before_aggregate_supported_features
            ]
        )
        logger.info("Aggregating...")
        data = self.aggregate(data)
        logger.info("Adding features after aggregation...")
        data = self.add_features(
            data,
            features=[
                feature for feature in self.features if feature["name"] in self.after_aggregate_supported_features
            ]
        )
        logger.info("Encoding...")
        data = self.encode(data)
        logger.info("Dropping unnecessary rows and columns...")
        data = self.drop(data)
        logger.info("Done...")

        assert data.isna().sum().sum() == 0, "NaN values found after transforming your data."

        logger.info("The transformed data looks like:")
        logger.info(data.info())

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

        data["date_block_num"] = data["date_block_num"].astype(np.int16)
        data["shop_id"] = data["shop_id"].astype(np.int16)
        data["shop_name"] = data["shop_name"].astype(pd.StringDtype())
        data["item_id"] = data["item_id"].astype(np.int32)
        data["item_name"] = data["item_name"].astype(pd.StringDtype())
        data["item_category_id"] = data["item_category_id"].astype(np.int16)
        data["item_category_name"] = data["item_category_name"].astype(pd.StringDtype())
        data["item_price"] = data["item_price"].astype(np.float32)
        data["item_cnt_day"] = data["item_cnt_day"].astype(np.int32)

        return data

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        agg_operations = dict(
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
        if "revenue" in self.dict_features:
            agg_operations["item_revenue"] = ("item_revenue_day", "sum")

        data = data.\
            groupby(["date_block_num", "shop_id", "item_id"], as_index=False).\
            agg(**agg_operations)
        data["date"] = data["date"].apply(lambda date: date.replace(day=1))

        # Get references to different ids & string features.
        unique_item_features = data[
            ["item_id", "item_name", "item_category_id", "item_category_name"]
        ].drop_duplicates()
        unique_datetime_features = data[["date_block_num", "date"]].drop_duplicates()
        unique_shop_features = data[["shop_id", "shop_name"]].drop_duplicates()

        # Keep a reference of the initial dtypes to cast them back after filling the missing values.
        dtypes = data.dtypes.to_dict()

        # Fill data discontinuity.
        # TODO: Make the filling on the whole "date_block_num" range?
        #  Not only from the initial date_block_num an item_id appeared.
        filled_data = []
        for block_num in data["date_block_num"].unique():
            block_num_shop_ids = data.loc[data["date_block_num"] == block_num, "shop_id"].unique()
            block_num_item_ids = data.loc[data["date_block_num"] == block_num, "item_id"].unique()

            possible_combinations = list(product(*[[block_num], block_num_shop_ids, block_num_item_ids]))
            filled_data.append(np.array(possible_combinations, dtype="int32"))

        filled_data = np.vstack(filled_data)
        filled_data = pd.DataFrame(filled_data, columns=["date_block_num", "shop_id", "item_id"])
        data = filled_data.merge(data, on=["date_block_num", "shop_id", "item_id"], how="left")

        # Merge unique features to fill missing ids & string values.
        data = data.drop(columns=["item_name", "item_category_id", "item_category_name"])
        data = data.merge(unique_item_features, on=["item_id"], how="left")
        data = data.drop(columns=["date"])
        data = data.merge(unique_datetime_features, on=["date_block_num"], how="left")
        data = data.drop(columns=["shop_name"])
        data = data.merge(unique_shop_features, on=["shop_id"], how="left")

        # Fill missing numerical values (e.g. price, sales, revenue).
        numeric_columns = data.select_dtypes(include=['number']).columns
        numeric_columns = set(numeric_columns)
        numeric_columns = numeric_columns - {"date_block_num", "shop_id", "item_id", "item_category_id"}
        numeric_columns = list(numeric_columns)
        data[numeric_columns] = data[numeric_columns].fillna(0)

        data = data.astype(dtypes)

        return data

    def add_features(self, data: pd.DataFrame, features: List[dict]) -> pd.DataFrame:
        for feature_config in (pbar := tqdm(features)):
            feature_name = feature_config["name"]
            feature_parameters = feature_config.get("parameters", {})
            pbar.set_description(f"Adding feature '{feature_name}'")

            f = self.supported_features[feature_name]
            data = f(data, **feature_parameters)

        return data

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.categorical_features is not None and len(self.categorical_features) > 0:
            categorical_features = [f for f in self.categorical_features if f in data.columns]
            data = pd.get_dummies(data, columns=categorical_features)

        return data

    def drop(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.drop_rows:
            max_lag = self._find_max_lag()
            logger.info(f"Dropping rows with 'date_block_num' > {max_lag}")
            data = data[data["date_block_num"] >= max_lag]

        if self.drop_columns is not None and len(self.drop_columns) > 0:
            drop_columns = [column for column in self.drop_columns if column in data.columns]
            data = data.drop(drop_columns, axis=1)

        return data

    def _find_max_lag(self) -> int:
        lags = self.dict_features.get("lags")
        if lags is None:
            return 0

        max_lag = 0
        lags = lags.get("parameters", {})
        for lagged_values in lags.values():
            column_max_lag = max(lagged_values["lags"])
            if column_max_lag > max_lag:
                max_lag = column_max_lag

        return max_lag

    @classmethod
    def _add_revenue_feature(cls, data: pd.DataFrame) -> pd.DataFrame:
        data["item_revenue_day"] = data["item_price"] * data["item_cnt_day"]
        data["item_revenue_day"] = data["item_revenue_day"].astype(np.float32)

        return data

    @classmethod
    def _add_time_feature(cls, data: pd.DataFrame) -> pd.DataFrame:
        @lru_cache(maxsize=128)
        def get_percentage_of_days(date, validation_function):
            start_date = date.replace(day=1)
            end_date = start_date + datetime.timedelta(days=date.days_in_month - 1)
            month_dates = pd.date_range(start=start_date, end=end_date, freq="D")

            valid_days = [1 if validation_function(d) else 0 for d in month_dates]
            valid_days = sum(valid_days)
            percentage_valid_days = valid_days / len(month_dates)

            return percentage_valid_days

        data["month"] = data["date"].apply(lambda date: date.month)
        data["business_days_percentage"] = data["date"].apply(
            func=partial(get_percentage_of_days, validation_function=utils.is_business_day)
        )
        data["holidays_percentage"] = data["date"].apply(
            func=partial(get_percentage_of_days, validation_function=utils.is_holiday)
        )

        data["month"] = data["month"].astype("int8")
        data["business_days_percentage"] = data["business_days_percentage"].astype("float16")
        data["holidays_percentage"] = data["holidays_percentage"].astype("float16")

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
        data.loc[data.city_name == "!????????????", "city_name"] = "????????????"
        data["city_id"] = LabelEncoder().fit_transform(data["city_name"])

        coords = dict()
        coords["????????????"] = (62.028098, 129.732555, 4)
        coords["????????????"] = (44.609764, 40.100516, 3)
        coords["????????????????"] = (55.8094500, 37.9580600, 1)
        coords["????????????????"] = (53.4305800, 50.1190000, 3)
        coords["??????????????"] = (59.2239000, 39.8839800, 2)
        coords["??????????????"] = (51.6720400, 39.1843000, 3)
        coords["????????????????"] = (0, 0, 0)
        coords["??????????????????"] = (55.5952800, 38.1202800, 1)
        coords["????????????????-??????????????"] = (0, 0, 0)
        coords["????????????"] = (55.7887400, 49.1221400, 4)
        coords["????????????"] = (54.5293000, 36.2754200, 4)
        coords["??????????????"] = (55.0794400, 38.7783300, 4)
        coords["????????????????????"] = (56.0183900, 92.8671700, 4)
        coords["??????????"] = (51.7373300, 36.1873500, 3)
        coords["????????????"] = (55.7522200, 37.6155600, 1)
        coords["????????????"] = (55.9116300, 37.7307600, 1)
        coords["??.????????????????"] = (56.3286700, 44.0020500, 4)
        coords["??????????????????????"] = (55.0415000, 82.9346000, 4)
        coords["????????"] = (54.9924400, 73.3685900, 4)
        coords["????????????????????????"] = (47.2313500, 39.7232800, 3)
        coords["??????"] = (59.9386300, 30.3141300, 2)
        coords["????????????"] = (53.2000700, 50.1500000, 4)
        coords["??????????????"] = (56.3000000, 38.1333300, 4)
        coords["????????????"] = (61.2500000, 73.4166700, 4)
        coords["??????????"] = (56.4977100, 84.9743700, 4)
        coords["????????????"] = (57.1522200, 65.5272200, 4)
        coords["??????"] = (54.7430600, 55.9677900, 4)
        coords["??????????"] = (55.8970400, 37.4296900, 1)
        coords["????????????????"] = (0, 0, 0)
        coords["??????????"] = (55.1477000, 37.4772800, 4)
        coords["??????????????????"] = (57.6298700, 39.8736800, 2)

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
                data["category_company_average_item_sales"] = \
                    data["category_company_average_item_sales"].fillna(0).astype("int16")
                data["category_company_average_item_price"] = \
                    data["category_company_average_item_price"].fillna(0).astype("float32")

            elif level == "city":
                level_category_df = data.groupby(["date_block_num", "city_id", "item_category_id"], as_index=False).agg(
                    category_city_average_item_sales=("item_sales", "mean"),
                    category_city_average_item_price=("item_price", "mean")
                )
                data = data.merge(level_category_df, on=["date_block_num", "city_id", "item_category_id"], how="left")
                data["category_city_average_item_sales"] = \
                    data["category_city_average_item_sales"].fillna(0).astype("int16")
                data["category_city_average_item_price"] = \
                    data["category_city_average_item_price"].fillna(0).astype("float32")
            else:
                level_category_df = data.groupby(["date_block_num", "shop_id", "item_category_id"], as_index=False).agg(
                    category_shop_average_item_sales=("item_sales", "mean"),
                    category_shop_average_item_price=("item_price", "mean")
                )
                data = data.merge(level_category_df, on=["date_block_num", "shop_id", "item_category_id"], how="left")
                data["category_shop_average_item_sales"] = \
                    data["category_shop_average_item_sales"].fillna(0).astype("int16")
                data["category_shop_average_item_price"] = \
                    data["category_shop_average_item_price"].fillna(0).astype("float32")

        return data

    @classmethod
    def _add_multiple_lag_feature(
            cls,
            data: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        for column, column_lagged_params in kwargs.items():
            lags = column_lagged_params["lags"]
            fill_value = column_lagged_params["fill_value"]

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
