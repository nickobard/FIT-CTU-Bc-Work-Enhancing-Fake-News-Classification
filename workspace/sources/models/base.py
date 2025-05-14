from abc import ABC
import mlflow
import pickle
import os
import utils
from ..experiments.metrics import Loss
from utils import generate_random_state
from ..utils import create_and_get_local_logger
from pathlib import Path
from ..utils import class_name_to_str, dict_signature


class Model(ABC):
    def __init__(self, train_best_model_metric=Loss):
        self.train_best_model_metric = train_best_model_metric
        self.random_state = None
        self.logger = None
        self.artifacts_path = None

    def init(self, logger=None, random_state=None):
        self.artifacts_path = self.get_artifacts_path()
        Path(self.artifacts_path).mkdir(parents=False, exist_ok=True)
        self.logger = logger if logger else create_and_get_local_logger(self.__class__.__name__)
        self.random_state = random_state if random_state else generate_random_state()
        return self

    def assemble_signature(self):
        model_params = self._params()
        model_params_signature = dict_signature(model_params)
        return f"model({model_params_signature})"

    def _params(self):
        return {}

    @staticmethod
    def get_artifacts_path():
        artifacts_path = utils.get_current_run_artifacts_path()
        if artifacts_path is None:
            return None
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
