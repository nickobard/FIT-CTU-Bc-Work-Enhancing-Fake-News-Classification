from ...experiments.metrics import Loss
from ..base import Model
import mlflow


class TransformersModels(Model):
    def __init__(self, train_best_model_metric=Loss):
        super().__init__(train_best_model_metric)

    @classmethod
    def load_from_mlflow(cls, logger):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {cls.__name__} class.")

    def save_to_mlflow(self):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {self.__class__.__name__} class.")
