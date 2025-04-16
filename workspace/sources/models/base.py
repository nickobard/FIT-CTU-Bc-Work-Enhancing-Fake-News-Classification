from abc import ABC
import mlflow
import pickle
import os
import utils
from experiments.metrics import FalsePositiveRate
from utils import generate_random_state


class Model(ABC):
    def __init__(self):
        self.main_metric = FalsePositiveRate()
        self.random_state = None
        self.logger = None

    class Builder:
        def __init__(self):
            self.model_class = Model
            self._logger = None
            self._random_state = generate_random_state()
            self._main_metric = FalsePositiveRate()

        def with_logger(self, logger):
            self._logger = logger
            return self

        def with_random_state(self, random_state):
            self._random_state = random_state
            return self

        def with_main_metric(self, main_metric):
            self._main_metric = main_metric
            return self

        def build(self):
            self._logger.debug(f"Building model class: {self.model_class}")
            model = self.model_class()
            model.logger = self._logger
            model.random_state = self._random_state
            model.main_metric = self._main_metric
            return model

    @staticmethod
    def get_model_artifacts_path():
        artifact_uri = mlflow.active_run().info.artifact_uri
        artifcats_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
        model_artifacts_path = os.path.join(artifcats_path, 'model')
        return model_artifacts_path

    @classmethod
    def load_from_mlflow(cls, logger):
        if cls.mlflow_model_artifact_exists(logger):
            model_artifacts_path = cls.get_model_artifacts_path()
            model_path = os.path.join(model_artifacts_path, 'model.pkl')
            logger.info(f"Attempting to load model artifact from {model_path}.")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model artifact loaded successfully.")
            model.set_logger(logger)
            return model
        else:
            raise FileNotFoundError(f"Could not load model from mlflow.")

    @classmethod
    def mlflow_model_artifact_exists(cls, logger):
        local_path = cls.get_model_artifacts_path()
        model_path = os.path.join(local_path, 'model.pkl')
        if os.path.exists(model_path):
            return True
        else:
            logger.error(f"Error: Model artifact not found at path: {model_path}")
            return False

    def save_to_mlflow(self):
        model_dir = self.get_model_artifacts_path()
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
