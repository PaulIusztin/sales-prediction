import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from datasets import Dataset
from models import Model


class LinearRegressionModel(Model):
    def __init__(self, hyper_parameters: Optional[dict] = None):
        from evaluators import RegressionEvaluator

        super().__init__("linear_regression", hyper_parameters=hyper_parameters)

        self._feature_names: Optional[List[str]] = None
        self._model: Optional[LinearRegression] = None
        self.evaluator = RegressionEvaluator()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self._feature_names = X_train.columns.values.tolist()

        self._model = LinearRegression(**self.hyper_parameters)
        self._model = self._model.fit(X_train, y_train)

        # TODO: See how to refactor this group of code between classes.
        if self.logger is not None:
            train_results = self.evaluator.compute(
                model=self,
                dataset=dataset,
                split="train",
                plot=False,
                output_dir=None
            )
            validation_results = self.evaluator.compute(
                model=self,
                dataset=dataset,
                split="validation",
                plot=False,
                output_dir=None
            )
            for split_name, results in (("train", train_results), ("validation", validation_results)):
                for metric_name, metric_value in results.items():
                    self.logger.report_scalar(
                        title=metric_name.upper(),
                        series=f"LinearRegression/{split_name}",
                        value=metric_value,
                        iteration=0
                    )

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        assert self._model is not None, "Model has not been trained yet."

        return self._model.predict(X)

    def save(self, output_path: str):
        assert self._model is not None, "Model has not been trained yet."

        if not output_path.endswith(".pickle"):
            output_path += ".pickle"

        with open(output_path, "wb") as f:
            pickle.dump(self._model, f)

    def plot(self, output_dir: str):
        assert self._model is not None, "Model has not been trained yet."

        coefficients = self._model.coef_
        intercept = self._model.intercept_
        feature_names = [*self._feature_names, "bias"]
        coefficients = np.concatenate([coefficients.flatten(), intercept])
        coefficients = {
            "coefficients": coefficients,
            "absolute_coefficients": np.abs(coefficients),
        }
        feature_importance_df = pd.DataFrame(data=coefficients, index=feature_names)
        feature_importance_df = feature_importance_df.sort_values(by="absolute_coefficients", ascending=True)

        feature_importance_df["coefficients"].plot.barh(figsize=(10, 12))
        plt.title(self.name)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "feature_importance.png")
        plt.clf()

        if self.logger is not None:
            self.logger.report_image(
                title=self.name,
                series="feature_importance",
                iteration=0,
                local_path=Path(output_dir) / "feature_importance.png"
            )
