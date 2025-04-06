from abc import ABC
from datasets import Dataset
from model import Model
class Experiment(ABC):
    class Builder:
        def __init__(self):
            self._params = {}

        def with_dataset(self, dataset):
            self._params["dataset"] = dataset
            return self

        def with_model(self, model):
            self._params["model"] = model
            return self

        def build(self):
            experiment = Experiment()
            for key, value in self._params.items():
                setattr(experiment, key, value)
            return experiment

    def __init__(self):
        self.dataset = None
        self.model = None

    def conduct(self):
        self.dataset.log()
        self.model.log()
