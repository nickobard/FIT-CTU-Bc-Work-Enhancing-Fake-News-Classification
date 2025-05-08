from abc import abstractmethod
import os
from pathlib import Path
import pandas as pd
import datasets as hf_datasets


class Data:
    @abstractmethod
    def to_csv(self, path):
        pass


class PandasData(Data):
    def __init__(self, dataset, features_column_name='article', labels_column_name='label'):
        self.features_colname = features_column_name
        self.labels_colname = labels_column_name
        self.dataset = dataset

    @property
    def features(self):
        return self.dataset[self.features_colname]

    @features.setter
    def features(self, value):
        self.dataset[self.features_colname] = value

    @property
    def labels(self):
        return self.dataset[self.labels_colname]

    @labels.setter
    def labels(self, value):
        self.dataset[self.labels_colname] = value

    def copy(self):
        return self.__class__(self.dataset.copy(),
                              self.features_colname,
                              self.labels_colname)

    @classmethod
    def load(cls, path):
        dataset = pd.read_csv(path)
        return cls(dataset)

    def to_csv(self, path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        self.dataset.to_csv(path, index=False)

    def is_empty(self):
        return len(self.dataset) == 0


class HuggingFaceData(Data):
    OUTPUT_FORMAT = '.csv'

    def __init__(self, dataset):
        self.dataset = dataset

    def to_csv(self, path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        self.dataset.to_csv(path, index=False)

    @classmethod
    def load(cls, path):
        dataset = pd.read_csv(path + cls.OUTPUT_FORMAT)
        hg_dataset = hf_datasets.Dataset.from_pandas(dataset)
        return cls(hg_dataset)
