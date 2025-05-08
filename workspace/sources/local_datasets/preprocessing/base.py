import os
import re
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path

from ..data_classes import PandasData


class Preprocessing(ABC):
    PREPROCESSED_DATA_CLASS = PandasData

    def __init__(self):
        self.logger = None

    @abstractmethod
    def preprocess(self, data):
        pass

    def init(self, logger=None):
        self.logger = logger if logger else getLogger()
        return self

    def hyperparameters_output(self):
        return {}

    def name(self):
        name = self.__class__.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

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
