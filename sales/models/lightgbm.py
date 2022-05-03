from pathlib import Path

import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt

from datasets import Dataset
from models.base import Model


class LightGBMModel(Model):
    def __init__(self, hyper_parameters: dict, meta_parameters: dict):
        super().__init__("lightgbm", hyper_parameters=hyper_parameters)

        self.meta_parameters = meta_parameters
        self.model = None

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")
        X_validation, y_validation = dataset.get(split="validation")

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_validation = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        evals_result = {}
        self.model = lgb.train(
            self.hyper_parameters,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_validation],
            feature_name=X_train.columns.tolist(),
            verbose_eval=100,
            evals_result=evals_result,
            early_stopping_rounds=self.meta_parameters.get("early_stopping_rounds"),
        )

        return self

    def plot(self, output_dir: str):
        lgb.plot_importance(
            self.model,
            figsize=(20, 50),
            height=0.7,
            importance_type="gain",
            max_num_features=50
        )
        plt.title(self.name)
        plt.savefig(Path(output_dir) / "feature_importance.png")

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X, *args, **kwargs)
