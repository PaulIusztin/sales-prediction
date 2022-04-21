import datetime
from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import is_business_day, is_holiday


def _add_item_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df["item_revenue_day"] = df["item_price"] * df["item_cnt_day"]

    return df


def _add_is_new_item_feature(df: pd.DataFrame) -> pd.DataFrame:
    is_new_items_df = df.groupby(["item_id"])["date"].min().reset_index()
    is_new_items_df["is_new_item"] = 1
    df = pd.merge(df, is_new_items_df[["date", "item_id", "is_new_item"]],
                  on=["date", "item_id"], how="left")
    df["is_new_item"] = df["is_new_item"].fillna(0)
    df["is_new_item"] = df["is_new_item"].astype("int8")

    return df


def _add_first_shop_transaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    is_first_shop_transaction_df = df.groupby(["shop_id", "item_id"])["date"].min().reset_index()
    is_first_shop_transaction_df["is_first_shop_transaction"] = 1
    df = pd.merge(df, is_first_shop_transaction_df[["date", "shop_id", "item_id", "is_first_shop_transaction"]],
                  on=["date", "shop_id", "item_id"], how="left")
    df["is_first_shop_transaction"] = df["is_first_shop_transaction"].fillna(0)
    df["is_first_shop_transaction"] = df["is_first_shop_transaction"].astype("int8")

    return df


def _add_city_features(df: pd.DataFrame) -> pd.DataFrame:
    df["city_name"] = df["shop_name"].apply(lambda x: x.split()[0].lower())
    df.loc[df.city_name == "!якутск", "city_name"] = "якутск"
    df["city_id"] = LabelEncoder().fit_transform(df["city_name"])

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

    df["city_coord_1"] = df["city_name"].apply(lambda x: coords[x][0])
    df["city_coord_2"] = df["city_name"].apply(lambda x: coords[x][1])
    df["country_part"] = df["city_name"].apply(lambda x: coords[x][2])

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Are those categorical?
    df["day_of_year"] = df["date"].apply(lambda date: date.day_of_year)
    df["day_of_month"] = df["date"].apply(lambda date: date.day)
    df["days_in_month"] = df["date"].apply(lambda date: date.days_in_month)
    # Everything starts from 1, except day_of_week. Map it to [1-7] for consistency.
    df["day_of_week"] = df["date"].apply(lambda date: date.day_of_week + 1)
    df["week_of_year"] = df["date"].apply(lambda date: date.weekofyear)
    df["month_of_year"] = df["date"].apply(lambda date: date.month)

    df["is_week_start"] = df["date"].apply(lambda date: (date.day_of_week + 1) == 1)
    df["is_week_end"] = df["date"].apply(lambda date: (date.day_of_week + 1) == 7)
    df["is_month_start"] = df["date"].apply(lambda date: date.is_month_start)
    df["is_month_end"] = df["date"].apply(lambda date: date.is_month_end)
    df["is_year_start"] = df["date"].apply(lambda date: date.is_year_start)
    df["is_year_end"] = df["date"].apply(lambda date: date.is_year_end)
    df["is_business_day"] = df["date"].apply(lambda date: is_business_day(date))
    df["is_holiday"] = df["date"].apply(lambda date: is_holiday(date))

    df["is_week_start"] = df["is_week_start"].astype("int8")
    df["is_week_end"] = df["is_week_end"].astype("int8")
    df["is_month_start"] = df["is_month_start"].astype("int8")
    df["is_month_end"] = df["is_month_end"].astype("int8")
    df["is_year_start"] = df["is_year_start"].astype("int8")
    df["is_year_end"] = df["is_year_end"].astype("int8")
    df["is_business_day"] = df["is_business_day"].astype("int8")
    df["is_holiday"] = df["is_holiday"].astype("int8")

    return df


def _add_average_category_sales(df: pd.DataFrame, levels: List[str] = ("company", "city", "shop")) -> pd.DataFrame:
    # TODO: Should we add this info only as lag / Is is data leakage?
    for level in levels:
        assert level in ("company", "city", "shop")

        if level == "company":
            level_category_df = df.groupby(["date", "item_category_id"], as_index=False). \
                agg({"item_cnt_day": ["mean"], "item_price": ["mean"]})
            level_category_df.columns = [
                "date", "item_category_id", "company_category_cnt_day", "company_category_item_price"
            ]
            df = df.merge(level_category_df, on=["date", "item_category_id"], how="left")
        elif level == "city":
            level_category_df = df.groupby(["date", "city_id", "item_category_id"], as_index=False). \
                agg({"item_cnt_day": ["mean"], "item_price": ["mean"]})
            level_category_df.columns = [
                "date", "city_id", "item_category_id", "city_category_cnt_day", "city_category_item_price"
            ]
            df = df.merge(level_category_df, on=["date", "city_id", "item_category_id"], how="left")
        else:
            level_category_df = df.groupby(["date", "shop_id", "item_category_id"], as_index=False). \
                agg({"item_cnt_day": ["mean"], "item_price": ["mean"]})
            level_category_df.columns = [
                "date", "shop_id", "item_category_id", "shop_category_cnt_day", "shop_category_item_price"
            ]
            df = df.merge(level_category_df, on=["date", "shop_id", "item_category_id"], how="left")

    return df


def _add_multiple_daily_lags(
        df: pd.DataFrame,
        lags: List[int],
        columns: List[str],
        fill_value: float = 0
) -> pd.DataFrame:
    for column in columns:
        df = _add_daily_lags(df, lags, column, fill_value)

    return df


def _add_daily_lags(df: pd.DataFrame, lags: List[int], column: str = "item_cnt_day", fill_value: float = 0):
    initial_dtype = df[column].dtype

    for lag in lags:
        lagged_df = df[["date", "shop_id", "item_id", column]].copy()
        lagged_df.columns = ["date", "shop_id", "item_id", f"{column}_lag_{lag}"]
        lagged_df["date"] = lagged_df["date"] + datetime.timedelta(days=lag)

        df = df.merge(lagged_df, how="left", on=["date", "shop_id", "item_id"])
        df = df.fillna(fill_value)
        df[f"{column}_lag_{lag}"] = df[f"{column}_lag_{lag}"].astype(initial_dtype)

    return df
