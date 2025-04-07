from abc import ABC
import mlflow

class Model(ABC):
    pass

class BertBasedUncased(Model):
    def __init__(self):
        self.name = "bert-base-uncased"

    def init_log(self):
        mlflow.log_param('model_name', self.name)

