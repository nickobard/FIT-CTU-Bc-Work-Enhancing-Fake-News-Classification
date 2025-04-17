import os
from logging import getLogger

import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from utils import generate_random_state


class Dataset(ABC):
    LABELS_MAPPING = {0: 'fake', 1: 'reliable'}

    class Data:
        def __init__(self, dataset, features_colname='article', labels_colname='label'):
            self.features = dataset[features_colname].copy()
            self.labels = dataset[labels_colname].copy()

        @property
        def dataset(self):
            return pd.concat([self.features, self.labels], axis=1)

        def is_empty(self):
            return len(self.dataset) == 0

    def __init__(self, data_path, preprocessings=None, train_pct=0.7, val_pct=0.15):
        self.data_path = data_path
        self.preprocessings = preprocessings
        self.artifacts_path = None
        self.logger = None
        self.random_state = None
        self.dataset = None
        self.train_set = self.val_set = self.test_set = None
        self.name = self.__class__.__name__.lower()

    def init(self, artifacts_path=None, logger=None, random_state=None):
        self.artifacts_path = artifacts_path if artifacts_path else os.getcwd()
        self.random_state = random_state if random_state else generate_random_state()
        self.logger = logger if logger else getLogger()
        self.set_name(self.__class__.__name__.lower())
        self.load_dataset().split(self.train_set, self.val_set).preprocess()
        return self

    def set_name(self, name):
        self.name = name
        mlflow.log_param('dataset_name', self.name)

    def load_dataset(self):
        self.dataset = pd.read_csv(self.data_path)
        mlflow.log_param('dataset_shape', self.dataset.shape)
        return self

    def preprocess(self):
        for preprocessing_fn in self.preprocessings:
            for set in [self.train_set, self.val_set, self.test_set]:
                set.features = preprocessing_fn(set.features)
        return self

    def split(self, train_pct=0.7, val_pct=0.15):
        mlflow.log_params({'train_pct': train_pct, 'val_pct': val_pct})
        test_pct = 1 - train_pct - val_pct
        train_set, rest = train_test_split(self.dataset, train_size=train_pct, random_state=self.random_state)
        val_ratio = val_pct / (val_pct + test_pct)
        val_set, test_set = train_test_split(rest, test_size=(1 - val_ratio), random_state=self.random_state)
        mlflow.log_params({'train_shape': train_set.shape,
                           'val_shape': val_set.shape,
                           'test_shape': test_set.shape})
        self.train_set = self.Data(train_set)
        self.val_set = self.Data(val_set)
        self.test_set = self.Data(test_set)
        return self

    def get_features_labels_split(self, set):
        return set['article'], set['label']


class ReCovery(Dataset):
    def init(self, artifacts_path=None, logger=None, random_state=None):
        super().init(artifacts_path, logger, random_state)
        self.set_name('ReCovery')
        return self


if __name__ == "__main__":
    recovery = ReCovery('workspace/sources/datasets/Recovery/recovery.csv').init().split()
    print('Dataset shape:', recovery.dataset.shape)
    print('Test:', recovery.test_set.shape)
    print('Validation:', recovery.val_set.shape)
    print('Test:', recovery.test_set.shape)
