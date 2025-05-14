import os
import pandas as pd
import mlflow
from abc import ABC
from sklearn.model_selection import train_test_split
from pathlib import Path

from .preprocessing.pipelines import PreprocessingPipeline
from .data_classes import PandasData
from ..utils import generate_random_state, log_params, get_normalized_path_from_artifact_uri, \
    create_and_get_local_logger, dict_signature, class_name_to_str, SIGNATURE_PART_SEPARATOR


class Dataset(ABC):
    LABELS_MAPPING = {0: 'fake', 1: 'reliable'}
    DEFAULT_DATA_OBJ_CLASS = PandasData

    def __init__(self, name, data_path, preprocessings_pipeline=None, train_pct=0.7, val_pct=0.15, resave=False):
        self.name = name
        self.resave = False
        self.data_path = data_path
        if preprocessings_pipeline is None:
            self.preprocessings_pipeline = PreprocessingPipeline(name='empty')
        else:
            self.preprocessings_pipeline = preprocessings_pipeline
        self.artifacts_path = None
        self.logger = None
        self.random_state = None
        self.dataset = None
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.train_set = self.val_set = self.test_set = None
        self.preprocessed_train_set = self.preprocessed_val_set = self.preprocessed_test_set = None

    def init(self, logger=None, random_state=None):
        self.logger = logger if logger else create_and_get_local_logger(self.__class__.__name__)
        log_params(self.get_dataset_params())
        self.logger.info(self.get_dataset_params())
        self.artifacts_path = self.get_artifacts_path()
        if self.artifacts_path:
            Path(self.artifacts_path).mkdir(parents=False, exist_ok=True)
        self.random_state = random_state if random_state else generate_random_state()
        self.prepare_dataset()
        return self

    def prepare_dataset(self):
        self.preprocessings_pipeline.log_params(logger=self.logger)
        if not self.resave and self.prepared_dataset_exists():
            self.logger.info('Prepared dataset exists, trying to load it.')
            self.load_prepared_dataset()
        else:
            self.logger.info('Prepared dataset does not exist, computing from scratch.')
            (self.load_dataset()
             .split(train_pct=self.train_pct, val_pct=self.val_pct)
             .preprocess()
             .save_prepared_datasets())
        return self

    def load_dataset(self):
        self.dataset = pd.read_csv(self.data_path)
        log_params({'dataset_shape': self.dataset.shape})
        return self

    def load_prepared_dataset(self):
        self.dataset = pd.read_csv(os.path.join(self.artifacts_path, 'dataset.csv'))
        splits = ['train_set', 'val_set', 'test_set']
        for split in splits:
            path = os.path.join(self.artifacts_path, split)
            data = self.DEFAULT_DATA_OBJ_CLASS.load(path)
            setattr(self, split, data)
        if self.preprocessings_pipeline.is_empty():
            preprocessed_data_class = self.DEFAULT_DATA_OBJ_CLASS
        else:
            last_preprocessor = self.preprocessings_pipeline[-1]
            preprocessed_data_class = last_preprocessor.PREPROCESSED_DATA_CLASS
        preprocessed_splits = [f'preprocessed_{filename}' for filename in splits]
        for split in preprocessed_splits:
            path = os.path.join(self.artifacts_path, split)
            data = preprocessed_data_class.load(path)
            setattr(self, split, data)
        return self

    def prepared_dataset_exists(self):
        if not self.artifacts_path:
            return False
        if not os.path.exists(os.path.join(self.artifacts_path, 'dataset.csv')):
            return False
        splits = ['train_set', 'val_set', 'test_set']
        for split in splits:
            path = os.path.join(self.artifacts_path, split)
            if not self.DEFAULT_DATA_OBJ_CLASS.saved_data_exists(path):
                return False
        preprocessed_splits = [f'preprocessed_{filename}' for filename in splits]
        if self.preprocessings_pipeline.is_empty():
            last_preprocessor_data_class = self.DEFAULT_DATA_OBJ_CLASS
        else:
            last_preprocessor = self.preprocessings_pipeline[-1]
            last_preprocessor_data_class = last_preprocessor.PREPROCESSED_DATA_CLASS
        for split in preprocessed_splits:
            path = os.path.join(self.artifacts_path, split)
            if not last_preprocessor_data_class.saved_data_exists(path):
                return False
        return True

    def save_prepared_datasets(self):
        if not self.artifacts_path:
            self.logger.info('artifacts path is none, skipping saving.')
            return self
        self.dataset.to_csv(os.path.join(self.artifacts_path, 'dataset.csv'), index=False)
        splits = ['train_set', 'val_set', 'test_set']
        preprocessed_splits = [f'preprocessed_{filename}' for filename in splits]
        for split in splits + preprocessed_splits:
            save_path = os.path.join(self.artifacts_path, split)
            data = getattr(self, split)
            data.save(save_path)
        return self

    def preprocess(self):
        splits = ['train_set', 'val_set', 'test_set']
        for split in splits:
            # initializing preprocessed data with copied original data
            setattr(self, f'preprocessed_{split}', getattr(self, split).copy())

        self.preprocessings_pipeline.init(logger=self.logger)
        for index, preprocessor in enumerate(self.preprocessings_pipeline):
            preprocessor.init(logger=self.logger)

            self.logger.info(f'Preprocessing function {index}: {preprocessor.name()} ')

            for split in splits:

                self.logger.info(f'Preprocessing split: {split}')

                split_to_preprocess = getattr(self, f'preprocessed_{split}')

                if self.artifacts_path and preprocessor.preprocessed_data_exists(self.artifacts_path,
                                                                                 split,
                                                                                 name_dir_prefix=str(index)):
                    self.logger.info('Preprocessed data exists, loading it.')
                    preprocessed_split_data_obj = preprocessor.load(self.artifacts_path,
                                                                    split,
                                                                    name_dir_prefix=str(index))
                else:
                    self.logger.info('Preprocessed data does not exist, computing from scratch.')
                    preprocessed_split_data_obj = preprocessor.preprocess(split_to_preprocess.copy())
                    if self.artifacts_path:
                        preprocessor.save_preprocessed_data(self.artifacts_path,
                                                            split,
                                                            preprocessed_split_data_obj,
                                                            name_dir_prefix=str(index))
                setattr(self, f'preprocessed_{split}', preprocessed_split_data_obj)
                log_params({f'preprocessed_{split}_shape': preprocessed_split_data_obj.shape})
        return self

    def split(self, train_pct=0.7, val_pct=0.15):
        log_params({'train_pct': train_pct, 'val_pct': val_pct})
        test_pct = 1 - train_pct - val_pct
        train_set, rest = train_test_split(self.dataset, train_size=train_pct, random_state=self.random_state)
        val_ratio = val_pct / (val_pct + test_pct)
        val_set, test_set = train_test_split(rest, test_size=(1 - val_ratio), random_state=self.random_state)
        log_params({'train_set_shape': train_set.shape,
                    'val_set_shape': val_set.shape,
                    'test_set_shape': test_set.shape})

        self.train_set = self.DEFAULT_DATA_OBJ_CLASS(features=train_set['article'], labels=train_set['label'], )
        self.val_set = self.DEFAULT_DATA_OBJ_CLASS(features=val_set['article'], labels=val_set['label'], )
        self.test_set = self.DEFAULT_DATA_OBJ_CLASS(features=test_set['article'], labels=test_set['label'], )
        return self

    @staticmethod
    def get_features_labels_split(set):
        return set['article'], set['label']

    def get_artifacts_path(self):
        if not mlflow.active_run():
            return None
        artifact_uri = mlflow.active_run().info.artifact_uri
        artifacts_path = get_normalized_path_from_artifact_uri(artifact_uri)
        return os.path.join(artifacts_path, 'dataset')

    def assemble_signature(self):
        dataset_params = self.get_dataset_params()
        dataset_params_signature = dict_signature(dataset_params)
        dataset_signature = f"dataset({dataset_params_signature})"
        preprocessing_pipeline_signature = self.preprocessings_pipeline.assemble_signature()
        return SIGNATURE_PART_SEPARATOR.join([dataset_signature, preprocessing_pipeline_signature])

    def get_dataset_params(self):
        return {'dataset_name': self.name.lower(),
                'train_pct': self.train_pct,
                'val_pct': self.val_pct}


if __name__ == "__main__":
    recovery = Dataset('ReCovery', 'workspace/sources/local_datasets/ReCOVery/recovery.csv').init().split()
    print('Dataset shape:', recovery.dataset.shape)
    print('Test:', recovery.test_set.shape)
    print('Validation:', recovery.val_set.shape)
    print('Test:', recovery.test_set.shape)
