from .base import Model
from .lightgbm import LightGBMModel
from .persistence import PersistenceModel
from .regression import LinearRegressionModel
from .xgboost import XGBoostModel


__all__ = ["Model", "LightGBMModel", "PersistenceModel", "LinearRegressionModel", "XGBoostModel", "model_registry"]

# TODO: Build a more flexible registry of models where models can be registered with a decorator.


model_registry = {
    "lightgbm": LightGBMModel,
    "persistence": PersistenceModel,
    "linear_regression": LinearRegressionModel,
    "xgboost": XGBoostModel,
}
