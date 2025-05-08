import pickle
from abc import abstractmethod
import os
from pathlib import Path
import pandas as pd
import datasets as hf_datasets


class Data:
    PICKLE_FILE_POSTFIX_EXTENSION = '_data_obj.pkl'
    CSV_FILE_POSTFIX_EXTENSION = '.csv'

    @property
    @abstractmethod
    def dataset(self):
        pass

    def to_csv(self, path):
        self.dataset.to_csv(path + self.CSV_FILE_POSTFIX_EXTENSION, index=False)

    def to_pickle(self, path):
        pickle_path = path + self.PICKLE_FILE_POSTFIX_EXTENSION
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
        return self

    def save(self, path):
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_pickle(path)
        self.to_csv(path)
        return self

    @classmethod
    def saved_data_exists(cls, path):
        pickle_file_exists = os.path.exists(path + cls.PICKLE_FILE_POSTFIX_EXTENSION)
        csv_file_exists = os.path.exists(path + cls.CSV_FILE_POSTFIX_EXTENSION)
        return pickle_file_exists and csv_file_exists

    @classmethod
    def load(cls, path):
        with open(path + cls.PICKLE_FILE_POSTFIX_EXTENSION, 'rb') as f:
            data = pickle.load(f)
        return data


class PandasData(Data):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    @property
    def dataset(self):
        return pd.concat([self.features, self.labels], axis=1)

    def copy(self):
        return self.__class__(self.features.copy(), self.labels.copy())

    def is_empty(self):
        return len(self.dataset) == 0


class HuggingFaceData(Data):

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset
