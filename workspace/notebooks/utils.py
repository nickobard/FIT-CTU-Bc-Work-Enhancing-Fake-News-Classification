import mlflow
from ..sources.experiments.utils import get_best_models_by_metrics
from ..sources.experiments.metrics import standard_evaluation_metrics
from ..sources.experiments.visualizations.plots import plot_roc_curve, plot_confusion_matrix
import ast
import pandas as pd
import os
from IPython.display import display, Markdown
from workspace.sources.experiments.metrics import Loss, standard_evaluation_metrics
from workspace.sources.experiments.utils import get_best_models_by_metrics
from workspace.sources.experiments.visualizations.tables import create_metrics_comparison_df, \
    export_best_models_to_latex
from workspace.sources.experiments.visualizations.utils import METRICS_PLOT_NAMES_MAPPING


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


def hparams_by_metric_to_latex_table(hparams_by_metric, dataset_name, columns):
    for metric, model_hparams in hparams_by_metric.items():
        df = pd.DataFrame(list(model_hparams.items()), columns=columns)
        output_dir = f'assets/{dataset_name}/model_hyperparameters/'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'model_hyperparameters_table_by_{metric.name}.tex')
        float_format_fn = lambda x: f'{x:.3f}' if isinstance(x, (float, int)) else str(x).replace('_', '\\_')
        latex_table = df.to_latex(index=False,
                                  escape=True,
                                  float_format=float_format_fn,
                                  column_format='|c||c|')
        # Add horizontal lines between each row
        latex_table = latex_table.replace('\\\\', '\\\\ \\midrule')
        with open(output_path, 'w') as f:
            f.write(latex_table)


def produce_experiment_model_hparams_tables(model_best_runs_by_metrics, dataset_name):
    hyperparameters_by_metric = dict()
    for metric in standard_evaluation_metrics:
        model_hparams_dict_str = model_best_runs_by_metrics[metric.name].data.params['model_input_hyperparameters']
        model_hparams_dict = ast.literal_eval(model_hparams_dict_str)

        hyperparameters_by_metric[metric] = model_hparams_dict

    hparams_by_metric_to_latex_table(hyperparameters_by_metric, dataset_name, columns=['Hyperparameter', 'Value'])


def produce_experiment_pipeline_hparams_tables(model_best_runs_by_metrics, dataset_name):
    hyperparameters_by_metric = dict()
    for metric in standard_evaluation_metrics:
        pipeline_hparams = dict()
        pipeline_name = model_best_runs_by_metrics[metric.name].data.params['preprocessing_pipeline_name']
        pipeline_hparams['pipeline_name'] = pipeline_name

        pipeline = ast.literal_eval(model_best_runs_by_metrics[metric.name].data.params['preprocessing_pipeline'])
        for step, preprocessing in enumerate(pipeline):
            pipeline_hparams[f'step_{step}'] = preprocessing

        hyperparameters_by_metric[metric] = pipeline_hparams

    hparams_by_metric_to_latex_table(
        hyperparameters_by_metric, dataset_name, columns=['Parameter', 'Value']
    )


def produce_experiment_confusion_matrices(model_best_runs_by_metrics, dataset_name):
    for metric in standard_evaluation_metrics:
        best_run = model_best_runs_by_metrics[metric.name]
        best_model_run_id = best_run.info.run_id
        run_artifacts_path = get_run_artifacts_path(best_model_run_id)
        plot_confusion_matrix(run_artifacts_path,
                              metric,
                              dataset_name=dataset_name,
                              show_plot=False)


def produce_experiment_roc_curves(model_best_runs_by_metrics, dataset_name):
    for metric in standard_evaluation_metrics:
        best_run = model_best_runs_by_metrics[metric.name]
        best_model_run_id = best_run.info.run_id
        run_artifacts_path = get_run_artifacts_path(best_model_run_id)
        plot_roc_curve(run_artifacts_path,
                       metric,
                       dataset_name=dataset_name,
                       show_plot=False)


def get_best_models_results(dataset_name, experiments, display=True):
    best_models_by_metrics = get_best_models_by_metrics(dataset_name, experiments)
    output_dir = os.path.join('assets', dataset_name, 'tables')
    for metric in standard_evaluation_metrics:
        if display:
            display(Markdown(f'### {METRICS_PLOT_NAMES_MAPPING[metric.name]}:'))

        best_models_df = create_metrics_comparison_df(metric,
                                                      best_models_by_metrics,
                                                      output_dir)
        latex_table = export_best_models_to_latex(metric,
                                                  best_models_df,
                                                  output_dir)
        if display:
            display(Markdown('#### Dataframe:'))
            display(best_models_df)
            display(Markdown('#### LaTeX Table:'))
            print(latex_table)
