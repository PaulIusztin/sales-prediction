import lightgbm

from matplotlib import pyplot as plt

from datasets.base import Dataset
from trainers.base import Trainer


class LightGBMTrainer(Trainer):
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

    def __init__(self, output_folder: str, dataset: Dataset):
        super().__init__("lightgbm", output_folder, dataset)

    def load_model(self):
        return lightgbm.LGBMRegressor(**self.HYPER_PARAMETERS)

    def train(self):
        x_train, y_train = self.data["train"]

        eval_set = [self.data["train"], self.data["validation"]]
        categorical_features = [c for c in self.dataset.CATEGORICAL_FEATURES if c in x_train.columns]

        self.model = self.model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            eval_metric=["rmse"],
            verbose=100,
            categorical_feature=categorical_features,
            early_stopping_rounds=self.META_FEATURES["early_stopping"]
        )

        lightgbm.plot_importance(
            self.model,
            figsize=(20, 50),
            height=0.7,
            importance_type="gain",
            max_num_features=50
        )
        plt.savefig(self.output_folder / "feature_importance.png")
