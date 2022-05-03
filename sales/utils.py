from typing import List, Dict, Any

import holidays
import pandas as pd
from omegaconf import DictConfig, ListConfig


def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


def is_holiday(date, country="ru"):
    assert country in ("ru", )

    return date in holidays.RU()


def to_consistent_types(d: dict) -> dict:
    d = {**d}

    for k, v in d.items():
        if isinstance(v, list):
            d[k] = tuple(v)
        elif isinstance(v, dict):
            d[k] = to_consistent_types(v)

    return d


def omega_conf_to_dict(conf: Any) -> Any:
    if isinstance(conf, ListConfig):
        return [omega_conf_to_dict(item) for item in conf]

    if not isinstance(conf, (ListConfig, DictConfig)):
        return conf

    d = {}
    for k, v in conf.items():
        if isinstance(v, ListConfig):
            d[k] = [omega_conf_to_dict(item) for item in v]
        elif isinstance(v, DictConfig):
            d[k] = omega_conf_to_dict(v)
        else:
            d[k] = v

    return d


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    columns = [col for col in columns if col in df.columns]
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


def remove_outliers_threshold(df: pd.DataFrame, columns: Dict[str, Dict[str, float]]) -> pd.DataFrame:
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
