import json
import os.path
from pathlib import Path
from typing import Optional, List

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics

from datasets import Dataset
from models import Model


class RegressionEvaluator:
    def compute(self, model: Model, dataset: Dataset, plot: bool = True, output_folder: Optional[str] = None) -> dict:
        X_test, y_test = dataset.get(split="test")
        y_pred = model.predict(X=X_test)

        r2 = metrics.r2_score(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        results = {
            "r2": r2,
            "rmse": rmse
        }
        for metric_name, metric_result in results.items():
            model.logger.report_scalar(
                title=metric_name.upper(),
                series=model.name,
                value=metric_result,
                iteration=1
            )

        if output_folder is not None:
            results_file = Path(output_folder) / "results.json"
            with open(results_file, "w") as f:
                str_results = {k: str(v) for k, v in results.items()}
                json.dump(str_results, f)

        if plot:
            assert output_folder is not None, "If you want to plot, you need to specify an output folder."

            self.plot(model, y_test, y_pred, output_folder=output_folder)

        return results

    def plot(self, model: Model, y_true, y_pred, output_folder: str):
        plt.figure(figsize=(10, 10))
        plt.ylabel("Predicted")
        sns.regplot(x=y_true, y=y_pred, fit_reg=True, scatter_kws={"s": 100}, )

        regplot_path = Path(output_folder) / "regplot.png"
        plt.title(model.name)
        plt.savefig(regplot_path)

        plt.clf()
