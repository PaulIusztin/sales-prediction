import os.path
from typing import Tuple

import numpy as np
import pandas as pd


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
