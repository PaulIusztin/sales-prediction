from typing import List, Dict

import numpy as np
import pandas as pd


def _cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _cast_if_exists(col: str, dtype: type) -> pd.DataFrame:
        if col in df.columns:
            df[col] = df[col].astype(dtype)

        return df

    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

    df = _cast_if_exists(col="date_block_num", dtype=np.int8)
    df = _cast_if_exists(col="shop_id", dtype=np.int8)
    df = _cast_if_exists(col="item_id", dtype=np.int16)
    df = _cast_if_exists(col="item_category_id", dtype=np.int8)
    df = _cast_if_exists(col="item_cnt_day", dtype=np.int16)
    df = _cast_if_exists(col="item_price", dtype=np.float32)

    return df


def _aggregate_twin_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if "item_cnt_day" not in df.columns:
        return df

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


def _remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    columns = {col for col in columns if col in df.columns}
    if len(columns) == 0:
        return df

    to_compare_df = df[columns]

    q1 = to_compare_df.quantile(0.25)
    q3 = to_compare_df.quantile(0.75)
    iqr = q3 - q1

    outliers_mask = ~((to_compare_df < (q1 - 1.5 * iqr)) | (to_compare_df > (q3 + 1.5 * iqr)))
    outliers_mask = outliers_mask.all(axis=1)
    df = df[outliers_mask]

    return df


def _remove_outliers_threshold(df: pd.DataFrame, columns: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    columns = {col: v for col, v in columns.items() if col in df.columns}
    if len(columns) == 0:
        return df

    outliers_mask = pd.Series([True] * df.shape[0])
    for column, thresholds in columns.items():
        to_compare_df = df[column]
        min_threshold = thresholds.get("min", 0)
        max_threshold = thresholds["max"]
        outliers_mask &= (to_compare_df > min_threshold) & (to_compare_df < max_threshold)

    return df[outliers_mask]
