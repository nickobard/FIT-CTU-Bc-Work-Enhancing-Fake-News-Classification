from abc import ABC
import mlflow

class Model(ABC):
    def __init__(self, random_state):
        self.random_state = random_state

class BertBasedUncased(Model):
    def __init__(self, random_state):
        self.name = "bert-base-uncased"
        mlflow.log_param('model_name', self.name)
        super().__init__(random_state)

    def fit(self, dataset):
        train, val, test = dataset.split()



