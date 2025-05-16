import mlflow
from .metrics import standard_evaluation_metrics
from .metrics import Loss


def pick_best_run_by_metric(runs, metric, second_metric=Loss):
    """Pick the best run based on given metric and its loss value."""
    metrics_keys = [f'test_{metric.name}_by_{metric.name}', f'test_{second_metric.name}_by_{metric.name}']
    metrics_greater_is_better = [metric.greater_is_better,
                                 second_metric.greater_is_better]
    metrics = zip(metrics_keys, metrics_greater_is_better)
    # Filter runs that have required metrics
    valid_runs = [run for run in runs
                  if all(key in run.data.metrics for key in metrics_keys)]

    if not valid_runs:
        return None

    # Define sort key function
    def _get_run_sort_key(run):
        return tuple(run.data.metrics[m_key] if m_gib else -run.data.metrics[m_key] for m_key, m_gib in metrics)

    # Return run with best metrics
    return max(valid_runs, key=_get_run_sort_key)

def get_best_models_by_metrics(dataset_name, experiments, metrics = standard_evaluation_metrics):
    client = mlflow.tracking.MlflowClient()

    experiments_runs = dict()
    for experiment in experiments:
        experiment_runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name(experiment).experiment_id],
            filter_string=f"params.dataset_name = '{dataset_name}'"
        )
        experiments_runs[experiment] = experiment_runs

    best_models_by_metrics = dict()
    for metric in metrics:
        best_models_by_metrics[metric.name] = dict()
        for experiment, runs in experiments_runs.items():
            best_run = pick_best_run_by_metric(runs, metric)
            if best_run is not None:
                best_models_by_metrics[metric.name][experiment] = best_run
    return best_models_by_metrics