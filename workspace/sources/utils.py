import logging
import os
import re
from urllib.parse import urlparse
from urllib.request import url2pathname
import mlflow
from mlflow import MlflowException
from inspect import getsource
from IPython.display import HTML


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
    # split before Uppercase+lowercase sequences (e.g. “Base”)
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    # split before single uppercase (e.g. before “I” in “AI” or “G” in “GPT”)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def print_source(*functions):
    """Print the source code for the given function(s) or class(es)."""
    source_code = '\n\n'.join(getsource(fn) for fn in functions)
    try:
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight

        display(HTML(highlight(source_code, PythonLexer(), HtmlFormatter(full=True))))

    except ImportError:
        print(source_code)


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


def extract_signature(obj, is_key=False, short_keys=True):
    if isinstance(obj, list):
        return list_signature(obj, short_keys)
    elif isinstance(obj, dict):
        return dict_signature(obj, short_keys)
    else:
        obj_str_repr = str(obj)
        if short_keys and is_key:
            return str_short_signature(str(obj_str_repr))
        return str_signature(obj_str_repr)


def str_signature(str_, short=False):
    lowercased = str_.lower()
    if short:
        words = lowercased.split('_')
        return ''.join(word[0] for word in words)
    return lowercased


def str_short_signature(str_):
    return str_signature(str_, short=True)


def list_signature(list_, short_keys=True):
    list_of_signatures = [extract_signature(el, short_keys=short_keys) for el in list_]
    return f'[{SIGNATURE_SUBPART_SEPARATOR.join(list_of_signatures)}]'


def dict_signature(dict_, short_keys=True):
    def get_key_val_joined_signature(key, value):
        key_signature = extract_signature(key, is_key=True, short_keys=short_keys)
        val_signature = extract_signature(value, short_keys=short_keys)
        join = SIGNATURE_KEY_VALUE_SEPARATOR.join([key_signature, val_signature])
        return join

    key_val_signatures_joins = [
        get_key_val_joined_signature(key, value) for key, value in
        dict_.items()]
    signature = SIGNATURE_SUBPART_SEPARATOR.join(key_val_signatures_joins)
    return signature
