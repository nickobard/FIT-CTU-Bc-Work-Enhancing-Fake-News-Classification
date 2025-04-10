from abc import ABC
import mlflow
import pickle
import os
import utils


class Model(ABC):
    def __init__(self, random_state, logger):
        self.random_state = random_state
        self.logger = logger

    def set_logger(self, logger):
        self.logger = logger
        return self

    @classmethod
    def get_model_artifacts_path(cls):
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


if __name__ == "__main__":
    pass
