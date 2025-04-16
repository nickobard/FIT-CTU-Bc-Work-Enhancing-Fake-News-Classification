from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mlflow
import os


class Visualization(ABC):

    @abstractmethod
    def build_figure(self):
        pass

    def __init__(self, **kwargs):
        self.artifacts_path = None
        self.data = None
        self.logger = None
        self.set_visible = kwargs.get('logger', True)
        self.figure = None
        self.name = self.__class__.__name__.lower()

    def init(self, **kwargs):
        self.logger = kwargs.get('logger', None)
        self.data = kwargs.get('data', None)
        self.artifacts_path = kwargs.get('artifacts_path', None)
        return self

    def __del__(self):
        """Destructor to clean up resources such as figures."""
        if self.figure:
            plt.close(self.figure)

    def build_visualization(self):
        self.build_figure()

    def visualize(self):
        self.build_visualization()
        self.save_artifacts()
        self.show()

    def save_artifacts(self):
        if not self.artifacts_path:
            self.logger.debug('No artifacts path is set.')
            return
        save_path = os.path.join(self.artifacts_path, self.name)
        self.save_data_and_figures(save_path)

    def save_data_and_figures(self, save_path):
        self.save_figures(save_path)

    def save_figures(self, save_path):
        if self.figure is None:
            self.logger.debug(f'Figure is not built.')
            return
        full_path = save_path + '.png'
        self._delete_artifact(full_path)
        self.figure.savefig(full_path)

    def show(self):
        """Show the plot if set_visible is True."""
        if not self.set_visible:
            self.logger.debug(f"Visualization are set to be not visible.")
            return
        if not self.figure:
            self.logger.debug(f'Figure is not built.')
            return
        plt.show(self.figure)

    def _delete_artifact(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
            self.logger.debug(f"File deleted: {filename}")
        else:
            self.logger.debug(f"File does not exist: {filename}")
