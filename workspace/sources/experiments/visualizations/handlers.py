import experiments.visualizations.plots as plots
import experiments.visualizations.tables as tables


class VisualizationsHandler:
    def __init__(self, *visualizations):
        self.visualizations = visualizations
        self.logger = None
        self.artifacts_path = None

    def init(self, logger, artifacts_path):
        self.logger = logger
        self.artifacts_path = artifacts_path
        return self

    def handle_visualizations(self, data):
        for visualization in self.visualizations:
            visualization.init(data=data,
                               artifacts_path=self.artifacts_path,
                               logger=self.logger).visualize()


standard_visualizations_handler = VisualizationsHandler(
    tables.Metrics(),
    tables.Hyperparameters(),
    plots.ConfusionMatrix(),
    plots.ROC()
)
