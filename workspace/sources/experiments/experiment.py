from abc import ABC
import mlflow
import logging
from utils import generate_random_state
import os
from experiments.metrics import FalsePositiveRate


class Experiment(ABC):
    def __init__(self, logging_level=logging.INFO):
        self._init_logger(logging_level)
        self.name = None
        self.run_id = None
        self.model_builder = None
        self.dataset_class = None
        self.dataset_path = None
        self.random_state = None
        self.visualizations_handler = None

    class Builder:
        def __init__(self):
            self._name = None
            self._visualizations_handler = None
            self._dataset_class = None
            self._dataset_path = None
            self._model_builder = None
            self._random_state = generate_random_state()
            self._run_id = None
            self._logging_level = logging.INFO

        def with_name(self, name):
            self._name = name
            return self

        def with_visualizations_handler(self, visualizations_handler):
            self._visualizations_handler = visualizations_handler
            return self

        def with_dataset(self, dataset_class, dataset_path):
            self._dataset_class = dataset_class
            self._dataset_path = dataset_path
            return self

        def with_model(self, model_builder):
            self._model_builder = model_builder
            return self

        def with_random_state(self, random_state):
            self._random_state = random_state
            return self

        def with_run_id(self, run_id):
            self._run_id = run_id
            return self

        def with_logging_level(self, logging_level):
            self._logging_level = logging_level
            return self

        def build(self):
            experiment = Experiment(self._logging_level)
            experiment.run_id = self._run_id
            experiment.name = self._name
            experiment.dataset_class = self._dataset_class
            experiment.dataset_path = self._dataset_path
            experiment.model_builder = self._model_builder
            experiment.visualizations_handler = self._visualizations_handler
            experiment.random_state = self._random_state
            return experiment

    def _init_logger(self, logging_level):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _init_experiment(self):
        self.experiment_id = mlflow.set_experiment(self.name).experiment_id
        self.logger.info(f"MLflow experiment initialized with ID: {self.experiment_id}")

    def _prepare(self):
        self.dataset = self.dataset_class(self.dataset_path, self.random_state)
        self.dataset.set_logger(self.logger)
        self.model = (self.model_builder
                      .with_random_state(self.random_state)
                      .with_logger(self.logger)
                      .build())
        self.visualizations_handler.init(logger=self.logger,
                                         artifacts_path=self.model.get_model_artifacts_path())

    def run(self):
        self._init_experiment()
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id) as experiment_run:
            self._prepare()
            self.model.fit(self.dataset)
            self.model.evaluate(self.visualizations_handler)
