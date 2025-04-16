from experiments.metrics import FalsePositiveRate
from models.base import Model
import mlflow


class TransformersModels(Model):
    def __init__(self, main_metric=FalsePositiveRate()):
        super().__init__(main_metric)

    @classmethod
    def load_from_mlflow(cls, logger):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {cls.__name__} class.")

    def save_to_mlflow(self):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {self.__class__.__name__} class.")
