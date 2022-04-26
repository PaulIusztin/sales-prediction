from .base import Model
from .lightgbm import LightGBMModel
from .persistence import PersistenceModel


__all__ = ["Model", "LightGBMModel", "PersistenceModel", "model_registry"]

# TODO: Build a more flexible registry of models where models can be registered with a decorator.
model_registry = {
    "lightgbm": LightGBMModel,
    "persistence": PersistenceModel,
}
