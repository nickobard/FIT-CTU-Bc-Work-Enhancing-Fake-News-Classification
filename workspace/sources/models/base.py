from abc import ABC
import mlflow
import pickle
import os
import utils
from experiments.metrics import FalsePositiveRate
from utils import generate_random_state
from logging import getLogger
from pathlib import Path


class Model(ABC):
    def __init__(self, main_metric=FalsePositiveRate()):
        self.main_metric = main_metric
        self.random_state = None
        self.logger = None

    def init(self, logger=None, random_state=None):
        self.artifacts_path = self.get_artifacts_path()
        Path(self.artifacts_path).mkdir(parents=False, exist_ok=True)
        self.logger = logger if logger else getLogger()
        self.random_state = random_state if random_state else generate_random_state()
        return self

    @staticmethod
    def get_artifacts_path():
        artifact_uri = mlflow.active_run().info.artifact_uri
        artifacts_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
        return os.path.join(artifacts_path, 'model')

    @classmethod
    def load_from_mlflow(cls, logger):
        if cls.mlflow_model_artifact_exists(logger):
            model_artifacts_path = cls.get_artifacts_path()
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
        local_path = cls.get_artifacts_path()
        model_path = os.path.join(local_path, 'model.pkl')
        if os.path.exists(model_path):
            return True
        else:
            logger.error(f"Error: Model artifact not found at path: {model_path}")
            return False

    def save_to_mlflow(self):
        model_dir = self.get_artifacts_path()
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
