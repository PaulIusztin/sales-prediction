from functools import partial

import numpy as np
import pandas as pd
from statsmodels import api as sm


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
