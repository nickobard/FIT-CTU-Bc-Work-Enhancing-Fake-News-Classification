from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import mlflow
import os
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Visualization(ABC):
    def __init__(self):
        self.set_visible = None
        self.probabilities = None
        self.labels = None

    class Builder:
        def __init__(self, visualization_class):
            self.visualization_class = visualization_class
            self._set_visible = True
            self._logger = None
            self._probabilities = None
            self._labels = None
            self._artifacts_path = None

        def with_visible(self, visible: bool):
            self._set_visible = visible
            return self

        def with_logger(self, logger):
            self._logger = logger
            return self

        def with_probabilities(self, probabilities):
            self._probabilities = probabilities
            return self

        def with_labels(self, labels):
            self._labels = labels
            return self

        def with_artifacts_path(self, artifacts_path):
            self._artifacts_path = artifacts_path

        def build(self):
            self._logger.debug(f"Building visualization class: {self.visualization_class}")
            visualization = self.visualization_class()
            visualization.logger = self._logger
            visualization.set_visible = self._set_visible
            visualization.set_probabilities(self._probabilities)
            visualization.labels = self._labels
            visualization.artifacts_path = self._artifacts_path
            return visualization

    @staticmethod
    def save_and_log_artifact(file_path, description, figure=None):
        if figure:
            plt.savefig(file_path, bbox_inches="tight")
            plt.close(figure)
        mlflow.log_artifact(file_path)

    def _handle_visibility(self):
        """Show the plot if set_visible is True."""
        self.logger.debug(f"Visualization set_visible: {self.set_visible}")
        if self.set_visible:
            plt.show()

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
        return self

    @abstractmethod
    def visualize(self, model_artifact_path):
        pass


class VisualizationWithProbabilities(Visualization):
    def __init__(self):
        super().__init__()

    def set_probabilities(self, probabilities):
        super().set_probabilities(probabilities)
        self.predictions = (self.probabilities > 0.5).astype(int)
        return self


class ConfusionMatrix(VisualizationWithProbabilities):
    def __init__(self):
        super().__init__()

    def visualize(self):
        # Confusion matrix plot
        cm = confusion_matrix(self.labels, self.predictions)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap="Blues", alpha=0.8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
        plt.xlabel("Predictions")
        plt.ylabel("Actuals")
        cm_path = os.path.join(self.artifacts_path, "confusion_matrix.png")
        self._handle_visibility()
        self.save_and_log_artifact(cm_path, "Confusion Matrix", fig)


class ROC(Visualization):
    def __init__(self):
        super().__init__()

    def visualize(self):
        # ROC curve plot
        roc_auc_score_value = roc_auc_score(self.labels, self.probabilities)
        fpr, tpr, _ = roc_curve(self.labels, self.probabilities)
        fig = plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score_value)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(self.artifacts_path, "roc_curve.png")
        self._handle_visibility()
        self.save_and_log_artifact(roc_curve_path, "ROC Curve", fig)


class Metrics(VisualizationWithProbabilities):
    def __init__(self):
        super().__init__()

    def _calculate_metrics(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        precision = precision_score(self.labels, self.predictions)
        recall = recall_score(self.labels, self.predictions)
        f1 = f1_score(self.labels, self.predictions)
        roc_auc = roc_auc_score(self.labels, self.probabilities)
        cm = confusion_matrix(self.labels, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "False Positive Rate": false_positive_rate,
            "False Negative Rate": false_negative_rate
        }

    def create_metrics_dataframe(self, metrics):
        return pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })

    def visualize(self):
        metrics = self._calculate_metrics()
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
