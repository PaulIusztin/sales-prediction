from pathlib import Path

import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt

from datasets import Dataset
from models.base import Model


class LightGBMModel(Model):
    # TODO: Move this to a config file.
    # HYPER_PARAMETERS = {
    #     "num_leaves": 966,
    #     "cat_smooth": 45.01680827234465,
    #     "min_child_samples": 27,
    #     "min_child_weight": 0.021144950289224463,
    #     "max_bin": 214,
    #     "learning_rate": 0.01,
    #     "subsample_for_bin": 300000,
    #     "min_data_in_bin": 7,
    #     "colsample_bytree": 0.8,
    #     "subsample": 0.6,
    #     "subsample_freq": 5,
    #     "n_estimators": 8000,
    # }
    HYPER_PARAMETERS = {
        'objective': 'mse',
        'metric': 'rmse',
        'num_leaves': 2 ** 7 - 1,
        'learning_rate': 0.005,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'seed': 1,
        'verbose': 1
    }
    META_FEATURES = {
        "early_stopping": 30
    }

    def __init__(self):
        super().__init__("lightgbm")

        self.model = None

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        return cls()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")
        X_validation, y_validation = dataset.get(split="validation")

        # categorical_features = dataset.pipeline.CATEGORICAL_FEATURES
        # categorical_features = [c for c in categorical_features if c in X_train.columns]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_validation = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        evals_result = {}
        self.model = lgb.train(
            self.HYPER_PARAMETERS,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_validation],
            feature_name=X_train.columns.tolist(),
            # categorical_feature=categorical_features,
            verbose_eval=100,
            evals_result=evals_result,
            early_stopping_rounds=100
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
