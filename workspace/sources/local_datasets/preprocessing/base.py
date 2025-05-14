import os
import re
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path

from ..data_classes import PandasData
from ...utils import log_params, class_name_to_str, create_and_get_local_logger, dict_signature


class Preprocessing(ABC):
    PREPROCESSED_DATA_CLASS = PandasData

    def __init__(self):
        self.logger = None

    @abstractmethod
    def preprocess(self, data):
        pass

    def init(self, logger=None):
        self._set_logger(logger)
        return self

    def log_params(self, logger=None):
        self._set_logger(logger)
        exploded_params = {f'{self.name()}_param_{prep_key}': prep_val for prep_key, prep_val in
                           self._detailed_params().items()}
        params = {self.name(): self._detailed_params(),
                  **exploded_params}
        log_params(params, logger=self.logger)
        return params

    def name(self):
        name = self.__class__.__name__
        return class_name_to_str(name)

    def preprocessed_data_exists(self, artifacts_path, split_name=None, name_dir_prefix=''):
        if artifacts_path is None:
            return False
        if name_dir_prefix:
            name_dir = '_'.join([name_dir_prefix, self.name()])
        else:
            name_dir = self.name()
        path = os.path.join(artifacts_path, name_dir, split_name)
        return self.PREPROCESSED_DATA_CLASS.saved_data_exists(path)

    def load(self, artifacts_path, split_name, name_dir_prefix=''):
        if name_dir_prefix:
            name_dir = '_'.join([name_dir_prefix, self.name()])
        else:
            name_dir = self.name()
        path = os.path.join(artifacts_path, name_dir, split_name)
        data = self.PREPROCESSED_DATA_CLASS.load(path)
        return data

    def save_preprocessed_data(self, artifacts_path, split_name, data, name_dir_prefix=''):
        if name_dir_prefix:
            name_dir = '_'.join([name_dir_prefix, self.name()])
        else:
            name_dir = self.name()
        path = os.path.join(artifacts_path, name_dir, split_name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.save(path)

    def _detailed_params(self):
        return {**self._params()}

    def _params(self):
        return {}

    def assemble_signature(self):
        class_name = self.name()
        params_signature = dict_signature(self._params())
        signature = f'{class_name}({params_signature})'
        return signature

    def _set_logger(self, logger):
        if logger:
            self.logger = logger
        else:
            self.logger = self.logger if self.logger else create_and_get_local_logger(self.__class__.__name__)
        return self
