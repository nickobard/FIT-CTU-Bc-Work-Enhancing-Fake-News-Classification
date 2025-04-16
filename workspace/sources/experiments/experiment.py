from abc import ABC
import mlflow
import logging

from experiments.visualizations.handlers import standard_visualizations_handler
from utils import generate_random_state
import os
from experiments.metrics import FalsePositiveRate


class Experiment(ABC):
    def __init__(self, name, model, dataset_class, dataset_path, run_id=None, vis_handler=None, random_state=None,
                 logging_level=logging.INFO):
        self._init_logger(logging_level)
        self.name = name
        self.run_id = run_id
        self.model = model
        self.dataset_class = dataset_class
        self.dataset_path = dataset_path
        self.random_state = random_state if random_state else generate_random_state()
        self.vis_handler = vis_handler if vis_handler else standard_visualizations_handler

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
        self.model.init(logger=self.logger, random_state=self.random_state)
        self.vis_handler.init(logger=self.logger,
                              artifacts_path=self.model.get_model_artifacts_path())

    def run(self):
        self._init_experiment()
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id) as experiment_run:
            self._prepare()
            self.model.fit(self.dataset)
            self.model.evaluate(self.vis_handler)
