from pathlib import Path
from typing import Optional, List

import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt

from datasets import Dataset
from models.base import Model


class LightGBMModel(Model):
    def __init__(self, hyper_parameters: dict, meta_parameters: dict):
        super().__init__("lightgbm", hyper_parameters=hyper_parameters)

        self.meta_parameters = meta_parameters

        self._evaluation_results: Optional[dict] = None
        self._model: Optional[lgb.Booster] = None

    def _build_callbacks(self) -> List[callable]:
        self._evaluation_results = {}

        callbacks = [
            lgb.record_evaluation(
                eval_result=self._evaluation_results
            ), lgb.log_evaluation(
                period=self.meta_parameters.get("evaluation_period", 100)
            )
        ]
        if self.meta_parameters.get("early_stopping_rounds") is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.meta_parameters["early_stopping_rounds"]
                )
            )
        # It also has lgb.reset_parameter() to change hyper-parameters during training.

        return callbacks

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")
        X_validation, y_validation = dataset.get(split="validation")

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_validation = lgb.Dataset(X_validation, label=y_validation, reference=lgb_train)

        num_boost_round = self.hyper_parameters.pop("num_boost_round")
        self._model = lgb.train(
            self.hyper_parameters,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_validation],
            feature_name=X_train.columns.tolist(),
            callbacks=self._build_callbacks(),
        )

        if self.logger is not None:
            for split_name, metrics in self._evaluation_results.items():
                for metric_name, metric_values in metrics.items():
                    for i, metric_value in enumerate(metric_values):
                        self.logger.report_scalar(
                            title=metric_name.upper(),
                            series=f"LightGBM/{split_name}",
                            value=metric_value,
                            iteration=i
                        )

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        if hasattr(self._model, "best_iteration"):
            kwargs["num_iteration"] = self._model.best_iteration

        return self._model.predict(X, *args, **kwargs)

    def save(self, output_path: str):
        self._model.save_model(output_path)

    def plot(self, output_dir: str):
        lgb.plot_importance(
            self._model,
            figsize=(20, 50),
            height=0.7,
            importance_type="gain",
            max_num_features=50
        )
        plt.title(self.name)
        plt.savefig(Path(output_dir) / "feature_importance.png")
        plt.clf()
