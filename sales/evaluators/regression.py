import json
from pathlib import Path
from typing import Optional

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics

from datasets import Dataset
from models import Model


class RegressionEvaluator:
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

        r2 = metrics.r2_score(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        results = {
            "r2": r2,
            "rmse": rmse
        }

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
