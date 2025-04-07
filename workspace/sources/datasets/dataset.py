import pandas as pd
import mlflow
from abc import ABC, abstractmethod
class Dataset(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

class ReCoveryDataset(Dataset):
    def __init__(self, path):
        self.name = 'ReCovery'
        dataset = pd.read_csv(path)
        super().__init__(dataset)

    def init_log(self):
        mlflow.log_param('dataset_name', self.name)
