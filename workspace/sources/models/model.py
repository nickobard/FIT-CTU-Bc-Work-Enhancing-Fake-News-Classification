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
    def load_from_mlflow(cls, artifact_uri, logger):
        if cls.mlflow_model_artifact_exists(artifact_uri):
            local_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
            model_path = os.path.join(local_path, 'model', 'model.pkl')
            logger.info(f"Attempting to load model artifact from {model_path}.")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model artifact loaded successfully.")
            model.set_logger(logger)
            return model
        else:
            raise FileNotFoundError(f"Model artifact not found at: {os.path.join(artifact_uri, 'model', 'model.pkl')}")

    @classmethod
    def mlflow_model_artifact_exists(cls, artifact_uri, logger):
        local_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
        model_path = os.path.join(local_path, 'model', 'model.pkl')
        if os.path.exists(model_path):
            return True
        else:
            logger.error(f"Error: Model artifact not found at path: {model_path}")
            return False

    def save_to_mlflow(self):
        artifact_uri = mlflow.active_run().info.artifact_uri
        local_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
        model_dir = os.path.join(local_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    pass
