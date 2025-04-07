import pandas as pd
import mlflow
from abc import ABC, abstractmethod
class Dataset(ABC):
    def __init__(self, data: pd.DataFrame, random_state):
        self.random_state = random_state
        self.data = data

class ReCoveryDataset(Dataset):
    def __init__(self, path, random_state):
        self.name = 'ReCovery'
        mlflow.log_param('dataset_name', self.name)
        dataset = pd.read_csv(path)
        super().__init__(dataset, random_state)
