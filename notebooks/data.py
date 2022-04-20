import datetime
import json
import os.path
import shutil
from pathlib import Path
from typing import Tuple, List, Union, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from clean import _cast_columns, _aggregate_twin_transactions, _remove_outliers_iqr, _remove_outliers_threshold
from features import _add_item_revenue, _add_is_new_item_feature, _add_first_shop_transaction_feature, \
    _add_city_features, _add_time_features, _add_average_category_sales, _add_multiple_daily_lags
from stats import find_most_significant_acf_values, find_most_significant_pacf_values


def load_notebook(path_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    items_df = pd.read_csv(os.path.join(path_dir, "items.csv"))
    item_categories_df = pd.read_csv(os.path.join(path_dir, "item_categories.csv"))
    items_df = items_df.merge(item_categories_df, on="item_category_id")
    shops = pd.read_csv(os.path.join(path_dir, "shops.csv"))

    train_df = pd.read_csv(os.path.join(path_dir, "sales_train.csv"))
    train_df = train_df.merge(items_df, on="item_id", how="left")
    train_df = train_df.merge(shops, on="shop_id", how="left")

    test_df = pd.read_csv(os.path.join(path_dir, "test.csv"))
    test_df = test_df.merge(items_df, on="item_id", how="left")
    test_df = test_df.merge(shops, on="shop_id", how="left")
    test_df["item_cnt_month"] = np.nan

    return train_df, test_df


def load(path_dir: str = "../data") -> pd.DataFrame:
    items_df = pd.read_csv(os.path.join(path_dir, "items.csv"))
    item_categories_df = pd.read_csv(os.path.join(path_dir, "item_categories.csv"))
    items_df = items_df.merge(item_categories_df, on="item_category_id")
    shops = pd.read_csv(os.path.join(path_dir, "shops.csv"))

    df = pd.read_csv(os.path.join(path_dir, "sales_train.csv"))
    df = df.merge(items_df, on="item_id", how="left")
    df = df.merge(shops, on="shop_id", how="left")

    return df


def try_load_from_cache(
        features: Iterable[str],
        path_dir: str = "../outputs"
) -> Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    cache_dir = Path(path_dir) / "cache"
    if not cache_dir.exists():
        return None

    meta_file = cache_dir / "meta.json"
    if not meta_file.exists():
        return None
    with open(meta_file, "r") as f:
        try:
            meta = json.load(f)
        except ValueError:
            return None

    data_file = cache_dir / "data.feather"
    if not data_file.exists():
        return None
    df = pd.read_feather(data_file)
    df = df.drop(columns=["index"])

    # TODO: Implement issubset logic + take dynamically only the features that match.
    if not set(features) == set(meta["features"]):
        shutil.rmtree(cache_dir)  # Invalidate cache.
        return None

    if len(df) != meta["n_rows"]:
        shutil.rmtree(cache_dir)  # Invalidate cache.
        return None

    return df


def save_to_cache(df: pd.DataFrame, features: Iterable[str], path_dir: str = "../outputs") -> None:
    cache_dir = Path(path_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_file = cache_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump({"n_rows": len(df), "features": features}, f)

    data_file = cache_dir / "data.feather"
    df.reset_index().to_feather(data_file)


def pre_clean(
        df: pd.DataFrame,
        remove_outliers_by: str = "threshold"
) -> pd.DataFrame:
    """
    :param df: dataframe that will have
        * its NaN values filled
        * columns cast to the right type
        * outliers removed
        * other custom cleaning
    :param remove_outliers_by: "iqr" or "threshold"
    :return: cleaned dataframe
    """

    assert df.isna().sum().sum() == 0, "We expect a dataframe without NaN values."
    assert remove_outliers_by in ("iqr", "threshold"), "We expect a valid remove_outliers_by value."

    df = _cast_columns(df)
    df = df.sort_values(by=["date"])
    df = _aggregate_twin_transactions(df)
    if remove_outliers_by == "iqr":
        df = _remove_outliers_iqr(df, ["item_cnt_day", "item_price"])
    else:
        df = _remove_outliers_threshold(
            df, {
                "item_cnt_day": {"min": 0, "max": 1000},
                "item_price": {"min": 0, "max": 50000}
            }
        )

    return df


def add_features(df: pd.DataFrame, features: List[Dict[str, Union[str, dict]]]) -> pd.DataFrame:
    feature_functions = {
        "time": _add_time_features,
        "revenue": _add_item_revenue,
        "is_new_item": _add_is_new_item_feature,
        "is_first_shop_transaction": _add_first_shop_transaction_feature,
        "daily_lags": _add_multiple_daily_lags,
        "category_sales": _add_average_category_sales,
    }
    compatible_features = set(feature_functions.keys())

    df = _add_city_features(df)
    for feature in features:
        feature_name = feature["name"]
        assert feature_name in compatible_features, "We expect a valid feature name."

        feature_parameters = feature.get("parameters", {})
        df = feature_functions[feature_name](df, **feature_parameters)

    return df


def post_clean(
        x: pd.DataFrame,
        y: pd.DataFrame,
        x_drop_columns: Optional[Iterable[str]] = None,
        drop_from: datetime.datetime = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if drop_from is not None:
        mask = x["date"] >= drop_from
        x = x[mask]
        y = y[mask]
    if x_drop_columns is not None:
        x = x.drop(columns=x_drop_columns)

    return x, y


def split_features_labels(df: pd.DataFrame, label_columns: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df.drop(columns=label_columns)
    y = df[label_columns]

    return x, y


def split(x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    validation_month = 32
    test_month = 33

    train_mask = x["date_block_num"] < min(validation_month, test_month)
    x_train = x[train_mask]
    y_train = y[train_mask]

    validation_mask = x["date_block_num"] == validation_month
    x_validation = x[validation_mask]
    y_validation = y[validation_mask]

    test_mask = x["date_block_num"] == test_month
    x_test = x[test_mask]
    y_test = y[test_mask]

    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test)
    }


def scale(splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    return splits


if __name__ == "__main__":
    train_df, _ = load()
    train_df = pre_clean(train_df)
    train_df = add_features(
        train_df,
        features=[
            {
                "name": "revenue"
            },
            {
                "name": "category_sales"
            },
            {
                "name": "daily_lags",
                "parameters": {
                    "columns": ["item_cnt_day", "shop_category_cnt_day"],
                    "lags": [1, 2, 4, 6],
                    "fill_value": 0
                }
            }
        ]
    )
    print(train_df)

    print(find_most_significant_acf_values(train_df, shop_id=5))
    print(find_most_significant_pacf_values(train_df, shop_id=5))
