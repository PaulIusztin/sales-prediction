import json
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics as sk_metrics

from datasets import Dataset
from models import Model


class RegressionEvaluator:
    def __init__(self, metrics: Tuple[str] = ("r2", "rmse")):
        self.metrics = metrics

        self.supported_metrics = {
            "r2": sk_metrics.r2_score,
            "rmse": partial(sk_metrics.mean_squared_error, squared=False),
            "mse": sk_metrics.mean_squared_error,
        }

    def compute(
            self,
            model: Model,
            dataset: Dataset,
            split: str = "test",
            plot: bool = True,
            output_dir: Optional[str] = None
    ) -> dict:
        if model.use_scaled_data:
            X_test, y_test = dataset.get(split=split)
        else:
            X_test, y_test = dataset.get(split=split, scaled=False)
        y_pred = model.predict(X=X_test)

        results = {}
        for metric in self.metrics:
            f = self.supported_metrics[metric]
            results[metric] = f(y_test, y_pred)

        save_results = output_dir is not None
        if save_results:
            results_file = Path(output_dir) / "results.json"
            with open(results_file, "w") as f:
                str_results = {k: str(v) for k, v in results.items()}
                json.dump(str_results, f)

        if plot:
            assert output_dir is not None, "If you want to plot, you need to specify an output folder."

            self.plot(model, y_test, y_pred, output_dir=output_dir)

        return results

    def plot(self, model: Model, y_true, y_pred, output_dir: str):
        plt.figure(figsize=(10, 10))
        plt.ylabel("Predicted")
        sns.regplot(x=y_true, y=y_pred, fit_reg=True, scatter_kws={"s": 100}, )

        plt.title(model.name)
        plt.savefig(Path(output_dir) / "regplot.png")
        plt.clf()
