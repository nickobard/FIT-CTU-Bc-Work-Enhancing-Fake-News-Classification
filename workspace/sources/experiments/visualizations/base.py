from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mlflow


class Visualization(ABC):
    def __init__(self):
        self.logger = None
        self.set_visible = None
        self.probabilities = None
        self.labels = None

    class Builder:
        def __init__(self):
            self.visualization_class = Visualization
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
    def save_and_log_artifact(file_path, figure=None):
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
    def visualize(self):
        pass


class PredictionBased(Visualization, ABC):
    def __init__(self):
        self.predictions = None
        super().__init__()

    class Builder(Visualization.Builder):
        def __init__(self):
            self._predictions = None
            super().__init__()
            self.visualization_class = PredictionBased

        def with_probabilities(self, probabilities):
            super().with_probabilities(probabilities)
            self._predictions = (self._probabilities > 0.5).astype(int)
            return self

        def build(self):
            visualization = super().build()
            visualization.predictions = self._predictions
            return visualization
