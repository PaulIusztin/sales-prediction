from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt

from datasets import Dataset
from models import Model


class XGBoostModel(Model):
    def __init__(self, hyper_parameters: dict, meta_parameters: dict):
        super().__init__(
            "xgboost",
            hyper_parameters=hyper_parameters,
            meta_parameters=meta_parameters
        )

        self._evaluation_results: Optional[dict] = None
        self._model: Optional[xgb.Booster] = None

    def fit(self, dataset: Dataset) -> "Model":
        self._evaluation_results = {}

        X_train, y_train = dataset.get(split="train")
        X_validation, y_validation = dataset.get(split="validation")

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_validation = xgb.DMatrix(X_validation, label=y_validation)

        eval_list = [(d_validation, 'validation'), (d_train, 'train')]

        num_boost_round = self.hyper_parameters.pop("num_boost_round")
        self._model = xgb.train(
            self.hyper_parameters,
            d_train,
            num_boost_round=num_boost_round,
            evals=eval_list,
            evals_result=self._evaluation_results,
            early_stopping_rounds=self.meta_parameters.get("early_stopping_rounds"),
            verbose_eval=self.meta_parameters.get("evaluation_period")
        )

        if self.logger is not None:
            for split_name, metrics in self._evaluation_results.items():
                for metric_name, metric_values in metrics.items():
                    for i, metric_value in enumerate(metric_values):
                        self.logger.report_scalar(
                            title=metric_name.upper(),
                            series=f"XGBoost/{split_name}",
                            value=metric_value,
                            iteration=i
                        )

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model has not been trained yet."

        if hasattr(self._model, "best_iteration"):
            # It has the best_iteration attribute only if "early_stopping_rounds" was used when training.
            kwargs["iteration_range"] = (0, self._model.best_iteration + 1)

        d_test = xgb.DMatrix(X)
        predictions = self._model.predict(d_test, *args, **kwargs)
        predictions = pd.Series(predictions, index=X.index)

        return predictions

    def save(self, output_path: str):
        assert self._model is not None, "Model has not been trained yet."

        if not output_path.endswith(".model"):
            output_path += ".model"

        self._model.save_model(output_path)

    def plot(self, output_dir: str):
        assert self._model is not None, "Model has not been trained yet."

        _, ax = plt.subplots(figsize=(10, 12), dpi=80)
        xgb.plot_importance(
            self._model,
            height=0.75,
            importance_type="gain",
            max_num_features=50,
            show_values=False,
            ax=ax
        )
        plt.title(self.name)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "feature_importance.png")
        plt.clf()
