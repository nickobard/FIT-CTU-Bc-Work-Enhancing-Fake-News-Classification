from experiments.visualizations.plots import ROC, ConfusionMatrix
from experiments.visualizations.tables import Metrics


class VisualizationsHandler:
    def __init__(self, *visualizations):
        self.visualizations = visualizations
        self.logger = None
        self.artifacts_path = None

    def set_logger(self, logger):
        self.logger = logger
        return self

    def set_artifacts_path(self, artifacts_path):
        self.artifacts_path = artifacts_path
        return self

    def handle_visualizations(self, probabilities, labels):
        for visualization in self.visualizations:
            (visualization.with_logger(self.logger)
             .with_probabilities(probabilities)
             .with_labels(labels)
             .with_artifacts_path(self.artifacts_path))

            visualization_obj = visualization.build()
            visualization_obj.visualize()


standard_visualizations_handler = VisualizationsHandler(
    Metrics.Builder().with_visible(True),
    ConfusionMatrix.Builder().with_visible(True),
    ROC.Builder().with_visible(True)
)
