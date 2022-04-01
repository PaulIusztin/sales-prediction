import os.path
from typing import Tuple

import pandas as pd


def load_data(path_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    items_df = pd.read_csv(os.path.join(path_dir, "items.csv"))

    train_df = pd.read_csv(os.path.join(path_dir, "sales_train.csv"))
    train_df = train_df.merge(items_df, on="item_id", how="left")
    train_df = train_df.drop(["item_name"], axis=1)
    train_df["date"] = pd.to_datetime(train_df["date"], format="%d.%m.%Y")
    train_df = train_df.sort_values(by=["date"])

    test_df = pd.read_csv(os.path.join(path_dir, "test.csv"))
    test_df = test_df.merge(items_df, on="item_id", how="left")
    sample_submission_df = pd.read_csv(os.path.join(path_dir, "sample_submission.csv"))
    test_df = test_df.merge(sample_submission_df, on="ID", how="left")
    test_df = test_df.drop(["item_name"], axis=1)

    return train_df, test_df
