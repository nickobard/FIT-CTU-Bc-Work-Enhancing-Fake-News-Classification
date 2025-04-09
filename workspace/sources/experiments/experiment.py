from abc import ABC
from local_datasets.dataset import Dataset
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

        def with_model(self, model_class, load_model_from_mlflow_if_exists=True):
            self._params["model_class"] = model_class
            self._params["load_model_from_mlflow"] = load_model_from_mlflow_if_exists
            return self

        def with_random_state(self, random_state):
            self._params["random_state"] = random_state
            return self

        def with_run_id(self, run_id):
            self._params["run_id"] = run_id
            return self

        def build(self):
            if self._params.get('load_model_from_mlflow', False) and self._params.get("run_id", None) is None:
                raise ValueError("run_id must be provided when load_model_from_mlflow is True.")
            experiment = Experiment(**self._params)
            return experiment

    def __init__(self, **kwargs):
        self.dataset_class = kwargs['dataset_class']
        self.dataset_path = kwargs['dataset_path']
        self.model_class = kwargs['model_class']
        self.load_model_from_mlflow_if_exists = kwargs.get('load_model_from_mlflow_if_exists', False)
        self.run_id = kwargs.get('run_id', None)
        self.name = kwargs.get('name', self.model_class.__name__)
        self.random_state = kwargs.get('random_state', generate_random_state())

    def __init_experiment(self):
        self.experiment_id = mlflow.set_experiment(self.name).experiment_id

    def __prepare(self):
        self.dataset = self.dataset_class(self.dataset_path, self.random_state)
        if self.load_model_from_mlflow_if_exists and self.model_class.mlflow_model_artifact_exsits():
            self.model = self.model_class.load_from_mlflow(self.run_id)
        else:
            self.model = self.model_class(self.random_state)

    def run(self):
        self.__init_experiment()
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id) as experiment_run:
            self.__prepare()
            self.model.fit(self.dataset)
            self.model.save_to_mlflow()
            self.model.evaluate()
