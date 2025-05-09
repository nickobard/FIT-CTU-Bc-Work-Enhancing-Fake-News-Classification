from . import plots
from . import tables


class VisualizationsHandler:
    def __init__(self, *visualizations):
        self.visualizations = visualizations
        self.logger = None
        self.artifacts_path = None

    def init(self, logger):
        self.logger = logger
        return self

    def handle_visualizations(self, data):
        for visualization in self.visualizations:
            visualization.init(data=data,
                               logger=self.logger).visualize()


standard_visualizations_handler = VisualizationsHandler(
    tables.Metrics(),
    tables.Hyperparameters(),
    plots.ConfusionMatrix(),
    plots.ROC()
)
