import os
from logging import getLogger

import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from utils import generate_random_state


class Dataset(ABC):
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = None
        self.random_state = None
        self.data = None
        self.name = self.__class__.__name__.lower()

    def init(self, logger=None, random_state=None):
        self.random_state = random_state if random_state else generate_random_state()
        self.logger = logger if logger else getLogger()
        self.load_dataset()
        self.name = 'ReCovery'
        mlflow.log_param('dataset_name', self.name)
        return self

    def load_dataset(self):
        self.data = pd.read_csv(self.data_path)
        mlflow.log_param('dataset_shape', self.data.shape)


class ReCovery(Dataset):

    def split(self, train_pct=0.7, val_pct=0.15):
        mlflow.log_params({'train_pct': train_pct, 'val_pct': val_pct})
        test_pct = 1 - train_pct - val_pct
        train, rest = train_test_split(self.data, train_size=train_pct, random_state=self.random_state)
        val_ratio = val_pct / (val_pct + test_pct)
        val, test = train_test_split(rest, test_size=(1 - val_ratio), random_state=self.random_state)
        mlflow.log_params({'train_shape': train.shape,
                           'val_shape': val.shape,
                           'test_shape': test.shape})
        return train, val, test


if __name__ == "__main__":
    recovery = ReCovery('workspace/sources/datasets/Recovery/recovery.csv').init()
    t, v, s = recovery.split()
    print('Dataset shape:', recovery.data.shape)
    print('Test:', t.shape)
    print('Validation:', v.shape)
    print('Test:', s.shape)
