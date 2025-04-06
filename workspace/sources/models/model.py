from abc import ABC
import mlflow

class Model(ABC):
    pass

class BertBasedUncased(Model):
    def __init__(self):
        self.model_name = "bert-base-uncased"
        mlflow.log_params('model_name', self.model_name)

