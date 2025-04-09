import os

import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class Dataset(ABC):
    def __init__(self, data: pd.DataFrame, random_state):
        self.random_state = random_state
        self.data = data
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger
        return self


class ReCoveryDataset(Dataset):
    def __init__(self, path, random_state):
        self.name = 'ReCovery'
        mlflow.log_param('dataset_name', self.name)
        dataset = pd.read_csv(path)
        mlflow.log_param('dataset_shape', dataset.shape)
        super().__init__(dataset, random_state)

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
    recovery = ReCoveryDataset(path='workspace/sources/datasets/Recovery/recovery.csv', random_state=42)
    t, v, s = recovery.split()
    print('Dataset shape:', recovery.data.shape)
    print('Test:', t.shape)
    print('Validation:', v.shape)
    print('Test:', s.shape)
