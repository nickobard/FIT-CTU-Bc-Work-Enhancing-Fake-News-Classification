import pandas as pd
import mlflow
from abc import ABC, abstractmethod
class Dataset(ABC):
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset


class ReCoveryDataset(Dataset):
    def __init__(self, path='Recovery/recovery.csv'):
        self.dataset_name = 'ReCovery'

        mlflow.log_params('dataset_name', self.dataset_name)

        dataset = pd.read_csv(path)
        super().__init__(dataset)
