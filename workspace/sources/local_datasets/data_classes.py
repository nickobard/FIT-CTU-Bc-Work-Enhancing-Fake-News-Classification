import pickle
from abc import abstractmethod
import os
from pathlib import Path
import pandas as pd
import datasets as hf_datasets


class Data:

    @abstractmethod
    def to_csv(self, path):
        pass

    def to_pickle(self, path):
        pickle_path = path + '_data_obj.pkl'
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
        pickle_file_exists = os.path.exists(path + 'data_obj.pkl')
        csv_file_exists = os.path.exists(path + '.csv')
        return pickle_file_exists and csv_file_exists

    @classmethod
    def load(cls, path):
        with open(path + 'data_obj.pkl', 'rb') as f:
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

    def to_csv(self, path):
        self.dataset.to_csv(path + '.csv', index=False)

    def is_empty(self):
        return len(self.dataset) == 0


class HuggingFaceData(Data):
    OUTPUT_FORMAT = '.csv'

    def __init__(self, dataset):
        self.dataset = dataset

    def to_csv(self, path):
        self.dataset.to_csv(path + '.csv', index=False)

    @classmethod
    def load(cls, path):
        dataset = pd.read_csv(path + cls.OUTPUT_FORMAT)
        hg_dataset = hf_datasets.Dataset.from_pandas(dataset)
        return cls(hg_dataset)
