import os
import pickle
from logging import getLogger
import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from pathlib import Path
import utils
from utils import generate_random_state


class Dataset(ABC):
    LABELS_MAPPING = {0: 'fake', 1: 'reliable'}

    def __init__(self, name, data_path, preprocessings=None, train_pct=0.7, val_pct=0.15, resave=False):
        self.name = name
        self.resave = False
        self.data_path = data_path
        self.preprocessings = preprocessings
        self.artifacts_path = None
        self.logger = None
        self.random_state = None
        self.dataset = None
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.train_set = self.val_set = self.test_set = None
        self.preprocessed_train_set = self.preprocessed_val_set = self.preprocessed_test_set = None

    def init(self, logger=None, random_state=None):
        mlflow.log_param('dataset_name', self.name)
        self.artifacts_path = self.get_artifacts_path()
        Path(self.artifacts_path).mkdir(parents=False, exist_ok=True)
        self.random_state = random_state if random_state else generate_random_state()
        self.logger = logger if logger else getLogger()
        self.prepare_dataset()
        return self

    def prepare_dataset(self):
        if not self.resave and self.prepared_dataset_exist():
            self.load_prepared_dataset()
        else:
            (self.load_dataset()
             .split(train_pct=self.train_pct, val_pct=self.val_pct)
             .preprocess()
             .save_prepared_datasets())
        return self

    def load_dataset(self):
        self.dataset = pd.read_csv(self.data_path)
        mlflow.log_param('dataset_shape', self.dataset.shape)
        return self

    def load_prepared_dataset(self):
        self.dataset = pd.read_csv(os.path.join(self.artifacts_path, 'dataset.csv'))
        unprocessed_sets_filenames = ['train_set_data.pkl', 'val_set_data.pkl', 'test_set_data.pkl']
        processed_sets_filenames = [f'preprocessed_{filename}' for filename in unprocessed_sets_filenames]
        for pickle_file in unprocessed_sets_filenames + processed_sets_filenames:
            with open(os.path.join(self.artifacts_path, pickle_file), 'rb') as f:
                setattr(self, pickle_file[:-9], pickle.load(f))
        return self

    def prepared_dataset_exist(self):
        unprocessed_sets_filenames = ['train_set_data.pkl', 'val_set_data.pkl', 'test_set_data.pkl']
        processed_sets_filenames = [f'preprocessed_{filename}' for filename in unprocessed_sets_filenames]
        for filename in ['dataset.csv'] + unprocessed_sets_filenames + processed_sets_filenames:
            if not os.path.exists(os.path.join(self.artifacts_path, filename)):
                return False
        return True

    def save_prepared_datasets(self):
        self.dataset.to_csv(os.path.join(self.artifacts_path, 'dataset.csv'), index=False)
        unprocessed_sets_filenames = ['train_set_data.pkl', 'val_set_data.pkl', 'test_set_data.pkl']
        processed_sets_filenames = [f'preprocessed_{filename}' for filename in unprocessed_sets_filenames]
        for pickle_file in unprocessed_sets_filenames + processed_sets_filenames:
            with open(os.path.join(self.artifacts_path, pickle_file), 'wb') as f:
                pickle.dump(getattr(self, pickle_file[:-9]), f)
        return self

    def preprocess(self):
        splits = ['train_set', 'val_set', 'test_set']
        for split in splits:
            setattr(self, f'preprocessed_{split}', getattr(self, split).copy())
        for preprocessing_fn in self.preprocessings:
            for split in splits:
                split_to_preprocess = getattr(self, f'preprocessed_{split}')
                preprocessed_split = preprocessing_fn.preprocess(split_to_preprocess)
                setattr(self, f'preprocessed_{split}', preprocessed_split)
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

        self.train_set = PandasData(train_set)
        self.val_set = PandasData(val_set)
        self.test_set = PandasData(test_set)
        return self

    @staticmethod
    def get_features_labels_split(set):
        return set['article'], set['label']

    @staticmethod
    def get_artifacts_path():
        artifact_uri = mlflow.active_run().info.artifact_uri
        artifacts_path = utils.get_normalized_path_from_artifact_uri(artifact_uri)
        return os.path.join(artifacts_path, 'dataset')


class PandasData:
    def __init__(self, dataset, features_column_name='article', labels_column_name='label'):
        self.features_colname = features_column_name
        self.labels_colname = labels_column_name
        self.features = dataset[self.features_colname].copy()
        self.labels = dataset[self.labels_colname].copy()

    def copy(self):
        return self.__class__(self.dataset.copy(),
                              self.features_colname,
                              self.labels_colname)

    @property
    def dataset(self):
        return pd.concat([self.features, self.labels], axis=1)

    def is_empty(self):
        return len(self.dataset) == 0


class HuggingFaceData:
    def __init__(self, dataset, features_column_name='article', labels_column_name='label'):
        self.features_colname = features_column_name
        self.labels_colname = labels_column_name
        self.dataset = dataset


if __name__ == "__main__":
    recovery = Dataset('ReCovery', 'workspace/sources/datasets/Recovery/recovery.csv').init().split()
    print('Dataset shape:', recovery.dataset.shape)
    print('Test:', recovery.test_set.shape)
    print('Validation:', recovery.val_set.shape)
    print('Test:', recovery.test_set.shape)
