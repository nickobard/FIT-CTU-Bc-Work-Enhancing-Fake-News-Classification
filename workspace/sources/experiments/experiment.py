from abc import ABC
import mlflow
from mlflow.entities import ViewType
import logging
from ..utils import generate_random_state, log_params, create_and_get_local_logger, SIGNATURE_PART_SEPARATOR
import time


class Experiment(ABC):
    def __init__(self, name, model, dataset, run_id=None, random_state=None,
                 logging_level=logging.INFO):
        self._init_logger(logging_level)
        self.name = name
        self.run_id = run_id
        self.model = model
        self.dataset = dataset
        if run_id is None:
            self.random_state = random_state if random_state else generate_random_state()
        else:
            self.random_state = int(mlflow.get_run(run_id).data.params['random_state'])
        self.run_signature = None

    def _init_logger(self, logging_level):
        self.logger = create_and_get_local_logger(
            self.__class__.__name__, logging_level=logging_level
        )

    def _init_experiment(self):
        self.experiment_id = mlflow.set_experiment(self.name).experiment_id
        self.logger.info(f"MLflow experiment initialized with ID: {self.experiment_id}")
        self.run_signature = self.assemble_run_signature()

        existing_run = self._find_existing_run(self.run_signature)
        if existing_run:
            self.logger.info(f"Found existing run with signature {self.run_signature}")
            self.logger.info(f"Existing run ID: {existing_run.info.run_id}")
            return existing_run
        return None

    def _prepare(self, experiment_run):
        self.logger.info(f"Run ID: {experiment_run.info.run_id}")
        self.logger.info(f"Run name: {experiment_run.info.run_name}")
        log_params({'random_state': self.random_state,
                    'run_signature': self.run_signature}, logger=self.logger)
        self.dataset.init(logger=self.logger, random_state=self.random_state)
        self.model.init(logger=self.logger, random_state=self.random_state)

    def assemble_run_signature(self):
        model_signature = self.model.assemble_signature()
        self.logger.info(f"Model signature: {model_signature}")
        dataset_signature = self.dataset.assemble_signature()
        self.logger.info(f"Dataset signature: {dataset_signature}")
        run_signature = SIGNATURE_PART_SEPARATOR.join([model_signature,
                                                       dataset_signature])
        self.logger.info(f"Run signature: {run_signature}")
        return run_signature

    def assemble_run_name(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        random_state_signature = f'rs={self.random_state}'
        run_name = SIGNATURE_PART_SEPARATOR.join([self.run_signature,
                                                  timestamp,
                                                  random_state_signature])
        return run_name

    def _find_existing_run(self, run_signature):
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"params.run_signature = '{run_signature}'",
            run_view_type=ViewType.ACTIVE_ONLY
        )
        return runs[0] if runs else None

    def run(self):
        existing_run = self._init_experiment()
        if existing_run is not None:
            self.logger.info(f"Skipping current experiment run, because run with same signature already exists.")
            return

        with mlflow.start_run(run_id=self.run_id,
                              run_name=self.assemble_run_name(),
                              experiment_id=self.experiment_id) as experiment_run:
            self._prepare(experiment_run)
            self.model.fit(self.dataset)
            self.model.evaluate()

    def prune_artifacts(self):
        self.model.prune_artifacts()
        self.dataset.prune_artifacts()
