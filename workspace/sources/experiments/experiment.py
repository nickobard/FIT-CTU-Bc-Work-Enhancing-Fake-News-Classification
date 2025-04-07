from abc import ABC
from datasets.dataset import Dataset
from models.model import Model
import mlflow
from utils import generate_random_state

class Experiment(ABC):
    class Builder:
        def __init__(self):
            self._params = {}

        def with_name(self, name):
            self._params["name"] = name
            return self

        def with_dataset(self, dataset_class, dataset_path):
            self._params["dataset_class"] = dataset_class
            self._params["dataset_path"] = dataset_path
            return self

        def with_model(self, model_class):
            self._params["model_class"] = model_class
            return self

        def with_random_state(self, random_state):
            self._params["random_state"] = random_state
            return self

        def build(self):
            experiment = Experiment(**self._params)
            return experiment

    def __init__(self, **kwargs):
        self.dataset_class = kwargs['dataset_class']
        self.dataset_path = kwargs['dataset_path']
        self.model_class = kwargs['model_class']
        self.name = kwargs.get('name', self.model_class.__name__)
        self.random_state = kwargs.get('random_state', generate_random_state())

    def __init_experiment(self):
        self.mlflow_experiment = mlflow.set_experiment(self.name)
        self.experiment_id = self.mlflow_experiment.experiment_id

    def __prepare(self):
        self.dataset = self.dataset_class(self.dataset_path, self.random_state)
        self.model = self.model_class(self.random_state)

    def run(self):
        self.__init_experiment()
        with mlflow.start_run(experiment_id=self.experiment_id) as experiment_run:
            self.__prepare()
            pass
