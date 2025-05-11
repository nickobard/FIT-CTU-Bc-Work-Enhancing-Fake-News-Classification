import logging
import os
import re
from urllib.parse import urlparse
from urllib.request import url2pathname
import mlflow
from mlflow import MlflowException


def generate_random_state():
    return int.from_bytes(os.urandom(4), byteorder="big")


def get_normalized_path_from_artifact_uri(artifact_uri: str) -> str:
    parsed_path = urlparse(artifact_uri).path
    local_path = url2pathname(parsed_path)
    normalized_path = os.path.normpath(local_path)
    return normalized_path


def get_current_run_artifacts_path():
    if mlflow.active_run() is None:
        return None
    artifact_uri = mlflow.active_run().info.artifact_uri
    artifacts_path = get_normalized_path_from_artifact_uri(artifact_uri)
    return artifacts_path


def log_params(params: dict, logger=None, supress_warnings=True):
    active_run = mlflow.active_run()
    if active_run is not None:
        try:
            mlflow.log_params(params)
        except MlflowException as e:
            if not supress_warnings:
                logger.warning(e.message)
    else:
        logger = logger if logger else logging.getLogger()
        logger.info(f'mlflow is not active, could not log the params: {params}')


def log_metrics(metrics: dict, logger=None):
    if mlflow.active_run():
        mlflow.log_metrics(metrics)
    else:
        logger = logger if logger else logging.getLogger()
        logger.info(f'mlflow is not active, could not log the metrics: {metrics}')


def class_name_to_str(class_name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()


def create_and_get_local_logger(name, logging_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


SIGNATURE_PART_SEPARATOR = ';'
SIGNATURE_SUBPART_SEPARATOR = ','
SIGNATURE_KEY_VALUE_SEPARATOR = '='


def string_signature(string):
    words = string.split('_')
    return ''.join(word[0] for word in words)


def dict_signature(dict_):
    key_val_joins = [SIGNATURE_KEY_VALUE_SEPARATOR.join([string_signature(key), str(value).lower()]) for key, value in
                     dict_.items()]
    signature = SIGNATURE_SUBPART_SEPARATOR.join(key_val_joins)
    return signature
