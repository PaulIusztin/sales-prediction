import os.path
from functools import partial
from typing import Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


def load_data(path_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    items_df = pd.read_csv(os.path.join(path_dir, "items.csv"))
    item_categories_df = pd.read_csv(os.path.join(path_dir, "item_categories.csv"))
    items_df = items_df.merge(item_categories_df, on="item_category_id")
    shops = pd.read_csv(os.path.join(path_dir, "shops.csv"))

    train_df = pd.read_csv(os.path.join(path_dir, "sales_train.csv"))
    train_df = train_df.merge(items_df, on="item_id", how="left")
    train_df = train_df.merge(shops, on="shop_id", how="left")
    train_df["date"] = pd.to_datetime(train_df["date"], format="%d.%m.%Y")
    train_df = train_df.sort_values(by=["date"])

    test_df = pd.read_csv(os.path.join(path_dir, "test.csv"))
    test_df = test_df.merge(items_df, on="item_id", how="left")
    test_df = test_df.merge(shops, on="shop_id", how="left")
    test_df["item_cnt_month"] = np.nan

    return train_df, test_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe that will have its NaN values filled, useless columns dropped and outliers removed.
    :return: cleaned dataframe
    """

    # Data does not have any NaN values.
    # TODO: Should we keep all the columns?

    df = aggregate_twin_transactions(df)
    df = remove_outliers_iqr(df, columns=["item_price", "item_cnt_day"])

    return df


def aggregate_twin_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["date", "item_id", "shop_id"], as_index=False).agg({
        "item_price": ["mean"],
        "item_cnt_day": ["sum"],
        "date_block_num": ["first"],
        "item_name": ["first"],
        "item_category_id": ["first"],
        "item_category_name": ["first"],
        "shop_name": ["first"]
    })
    df.columns = [
        "date", "item_id", "shop_id", "item_price", "item_cnt_day", "date_block_num",
        "item_name", "item_category_id", "item_category_name", "shop_name"
    ]

    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    to_compare_df = df[columns]

    q1 = to_compare_df.quantile(0.25)
    q3 = to_compare_df.quantile(0.75)
    iqr = q3 - q1

    outliers_mask = ~((to_compare_df < (q1 - 1.5 * iqr)) | (to_compare_df > (q3 + 1.5 * iqr)))
    outliers_mask = outliers_mask.all(axis=1)
    df = df[outliers_mask]

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe that will have its features added.
    :return: dataframe with added features
    """

    feature_to_add_func = [
        _add_time_features,
        _add_item_revenue,
        _add_is_new_item_feature,
        _add_first_shop_transaction_feature,
        _add_city_features,
    ]

    for func in feature_to_add_func:
        df = func(df)

    return df


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
    # df["day_of_year"] = df["date"].apply(lambda date: date.day_of_year)
    # df["day_of_month"] = df["date"].apply(lambda date: date.day_of_year)
    # df["day_of_week"] = df["date"].apply(lambda date: date.day_of_year)
    # df["week_of_year"] = df["date"].apply(lambda date: date.week_of_year)
    # df["week_of_month"] = df["date"].apply(lambda date: date.week_of_year)
    # df["month_of_year"] = df["date"].apply(lambda date: date.month_of_year)

    return df


def find_most_significant_acf_values(df: pd.DataFrame, shop_id: int, keep: int = 12) -> np.ndarray:
    return find_most_significant_autocorrelation_values(
        df,
        shop_id,
        keep,
        correlation_method="acf"
    )


def find_most_significant_pacf_values(df: pd.DataFrame, shop_id: int, keep: int = 12) -> np.ndarray:
    return find_most_significant_autocorrelation_values(
        df,
        shop_id,
        keep,
        correlation_method="pacf"
    )


def find_most_significant_autocorrelation_values(
        df: pd.DataFrame,
        shop_id: int,
        keep: int,
        correlation_method: str
) -> np.ndarray:
    assert correlation_method in ("acf", "pacf")

    df = df.copy()
    df = df.query(f"shop_id == {shop_id}")
    df = df.groupby("date").sum()
    df = df.sort_index()

    if correlation_method == "acf":
        f = partial(sm.tsa.stattools.acf, missing="raise")
        nlags = len(df)
    else:
        f = partial(sm.tsa.stattools.pacf, method="ols")
        nlags = len(df) // 2 - 1

    assert nlags > 0
    corr, sign = f(df["item_cnt_day"], nlags=nlags, alpha=0.05)
    # Keep only points that have a statistical value different from 0.
    points = [
        (t, t_corr, t_sign) for t, (t_corr, t_sign) in enumerate(zip(corr, sign)) if
        t_corr > t_sign[0] - t_corr or t_corr < t_sign[1] - t_corr
    ]
    # Sort the points by their distance from the 95% significance threshold.
    points = sorted(points, key=lambda p: max(abs(p[1] - (p[2][0] - p[1])), abs(p[1] - (p[2][1] - p[1]))), reverse=True)
    points = [(t, t_corr) for (t, t_corr, _) in points[:keep]]
    points = np.array(points, dtype=np.float32)

    return points


if __name__ == "__main__":
    train_df, _ = load_data()
    train_df = clean(train_df)
    train_df = add_features(train_df)
    print(train_df)

    print(find_most_significant_acf_values(train_df, shop_id=5))
    print(find_most_significant_pacf_values(train_df, shop_id=5))
