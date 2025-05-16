import os
import pandas as pd

from .utils import METRICS_PLOT_NAMES_MAPPING
from ..metrics import standard_metrics


def export_best_models_to_latex(metric, metrics_df: pd.DataFrame,
                                output_dir: str = 'assets/tables/') -> str:
    by_metric = metric.name
    experiment_columns = ['Model']
    metrics_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR', 'FNR']
    selected_columns = experiment_columns + metrics_columns
    selection_df = metrics_df[selected_columns]

    # Add bold formatting to specific columns
    bold_columns = ['Accuracy', 'Precision', 'F1-Score', 'ROC-AUC', 'FPR']
    column_format_left_size = '|' + '|'.join(['l'] * len(experiment_columns)) + '|'
    column_format_right_size = '|' + '|'.join(['r'] * len(metrics_columns)) + '|'
    column_format = column_format_left_size + column_format_right_size

    float_format_fn = lambda x: f'{x:.3f}' if isinstance(x, (float, int)) else x

    latex_table = selection_df.to_latex(index=False, escape=False,
                                        float_format=float_format_fn,
                                        column_format=column_format)
    latex_table = latex_table.replace('\\midrule', '\\midrule \\midrule')
    for col in bold_columns:
        latex_table = latex_table.replace(col, f'\\textbf{{{col}}}')

    output_path = os.path.join(output_dir, f'best_models_table_by_{by_metric}.tex')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_table)

    return latex_table


def create_metrics_comparison_df(metric,
                                 df_data,
                                 output_dir='assets/tables/'):
    metrics_df_data = []
    experiments_data = df_data[metric.name]
    for experiment_name, run in experiments_data.items():
        experiment_data = {
            'Model': run.data.params['model_name']
        }
        metrics_data = {METRICS_PLOT_NAMES_MAPPING[m.name]: run.data.metrics[f'test_{m.name}_by_{metric.name}']
                        for m in standard_metrics
                        if f'test_{m.name}_by_{metric.name}' in run.data.metrics}
        additional_metrics_data = {
            'best_epoch': run.data.metrics[f'best_epoch_by_{metric.name}'],
        }
        additional_experiment_data = {'Experiment': experiment_name,
                                      'Run Signature': run.data.params['run_signature'],
                                      'Run ID': run.info.run_id,
                                      'Evaluation Metric': metric.name}
        data = {**experiment_data,
                **metrics_data,
                **additional_experiment_data,
                **additional_metrics_data}
        metrics_df_data.append(data)
    metrics_df = pd.DataFrame(metrics_df_data)
    output_path = os.path.join(output_dir, f'best_models_table_by_{metric.name}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df
