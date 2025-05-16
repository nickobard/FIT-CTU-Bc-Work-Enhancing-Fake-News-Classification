import mlflow
from ..sources.experiments.utils import get_best_models_by_metrics


def flatten_dict(d: dict, flatten_key: str) -> dict:
    return {
        outer_k: (inner[flatten_key] if isinstance(inner, dict) and flatten_key in inner else inner)
        for outer_k, inner in d.items()
    }


def get_run_artifacts_path(run_id):
    client = mlflow.tracking.MlflowClient()
    run_artifacts_path = client.download_artifacts(run_id=run_id, path='')
    return run_artifacts_path


def get_experiment_best_runs_by_metrics(experiment_name, dataset_name):
    experiments_best_runs = get_best_models_by_metrics(dataset_name, [experiment_name])
    return flatten_dict(experiments_best_runs,
                        flatten_key=experiment_name)
