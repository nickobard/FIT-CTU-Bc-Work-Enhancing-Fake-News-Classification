from experiments.visualizations.vizualizations import PredictionBased, Visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd
import mlflow


class Metrics(PredictionBased):
    def __init__(self):
        super().__init__()

    class MetricsBuilder(PredictionBased.Builder):
        def __init__(self):
            super().__init__()
            self.visualization_class = Metrics

        def with_metrics_function(self, with_metrics_function):
            self._with_metrics_function = with_metrics_function
            return self

        def build(self):
            visualization = super().build()
            visualization.metrics_function = self._with_metrics_function
            return visualization

    def create_metrics_dataframe(self, metrics):
        return pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })

    def visualize(self):
        metrics = self.metrics_function

        metrics_df = self.create_metrics_dataframe(metrics)

        # Save the metric files
        metrics_csv_path = os.path.join(self.artifacts_path, "metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        self.save_and_log_artifact(metrics_csv_path, "CSV file of metrics")

        # Save the metric visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(metrics_df.columns))))
        metrics_png_path = os.path.join(self.artifacts_path, "metrics_visualization.png")
        self._handle_visibility()
        self.save_and_log_artifact(metrics_png_path, "Metrics visualization", fig)
