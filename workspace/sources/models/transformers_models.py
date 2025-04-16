from models.base import Model
import mlflow


class TransformersModels(Model):
    def __init__(self):
        super().__init__()

    class Builder(Model.Builder):
        def __init__(self):
            super().__init__()
            self.model_class = TransformersModels

    @classmethod
    def load_from_mlflow(cls, logger):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {cls.__name__} class.")

    def save_to_mlflow(self):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {self.__class__.__name__} class.")
