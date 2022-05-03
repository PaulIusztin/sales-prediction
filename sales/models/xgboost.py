from pathlib import Path

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt

from datasets import Dataset
from models import Model


class XGBoostModel(Model):
    def __init__(self, hyper_parameters: dict):
        super().__init__("xgboost", hyper_parameters=hyper_parameters)

        self.model = xgb.XGBRegressor(**self.hyper_parameters)

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self.model = self.model.fit(X_train, y_train)

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X)

    def plot(self, output_dir: str):
        feature_important = self.model.get_booster().get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"])
        # data = data.sort_values(by="score", ascending=False)
        data = data.nlargest(50, columns="score", keep="all")
        data.plot.barh(figsize=(25, 15))

        plt.title(self.name)
        plt.savefig(Path(output_dir) / "feature_importance.png")
