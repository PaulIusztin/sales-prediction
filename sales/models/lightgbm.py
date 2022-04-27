from pathlib import Path

import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt

from datasets import Dataset
from models.base import Model


class LightGBMModel(Model):
    # TODO: Move this to a config file.
    HYPER_PARAMETERS = {
        "num_leaves": 966,
        "cat_smooth": 45.01680827234465,
        "min_child_samples": 27,
        "min_child_weight": 0.021144950289224463,
        "max_bin": 214,
        "learning_rate": 0.01,
        "subsample_for_bin": 300000,
        "min_data_in_bin": 7,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "subsample_freq": 5,
        "n_estimators": 8000,
    }
    META_FEATURES = {
        "early_stopping": 30
    }

    def __init__(self):
        super().__init__("lightgbm")

        self.model = lgb.LGBMRegressor(**self.HYPER_PARAMETERS)

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        return cls()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")
        X_validation, y_validation = dataset.get(split="validation")
        categorical_features = dataset.CATEGORICAL_FEATURES

        eval_set = [(X_train, y_train), (X_validation, y_validation)]
        categorical_features = [c for c in categorical_features if c in X_train.columns]

        self.model = self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100,
            eval_metric=["rmse"],
            categorical_feature=categorical_features,
            early_stopping_rounds=self.META_FEATURES["early_stopping"],
        )

        return self

    def plot(self, output_folder: str):
        lgb.plot_importance(
            self.model,
            figsize=(20, 50),
            height=0.7,
            importance_type="gain",
            max_num_features=50
        )
        plt.savefig(Path(output_folder) / "feature_importance.png")

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X, *args, **kwargs)
