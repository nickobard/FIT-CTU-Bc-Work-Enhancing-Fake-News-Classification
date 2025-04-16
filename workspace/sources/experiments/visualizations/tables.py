from envs.JetBrains.Lib.abc import abstractmethod

from experiments.visualizations.base import Visualization
import matplotlib.pyplot as plt
import os
import pandas as pd


class Table(Visualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table = None

    @abstractmethod
    def build_table(self):
        pass

    def build_visualization(self):
        self.build_table()
        self.build_figure()

    def save_data_and_figures(self, save_path):
        self.save_table(save_path)
        self.save_figures(save_path)

    def save_table(self, save_path):
        if self.table is None:
            self.logger.debug(f'Table is not built.')
            return
        self.table.to_csv(save_path + '.csv', index=False)


class Metrics(Table):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None

    def init(self, **kwargs):
        super().init(**kwargs)
        self.metrics = self.data['metrics']
        return self

    def build_table(self):
        self.table = pd.DataFrame({
            "Metric": list(self.metrics.keys()),
            "Value": list(self.metrics.values())
        })

    def build_figure(self):
        # Save the metric visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(cellText=self.table.values, colLabels=self.table.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(self.table.columns))))


class Hyperparameters(Table):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparameters = None

    def init(self, **kwargs):
        super().init(**kwargs)
        self.hyperparameters = self.data['hyperparameters']
        return self

    def build_table(self):
        self.table = pd.DataFrame({
            "Hyperparameter": list(self.hyperparameters.keys()),
            "Value": list(self.hyperparameters.values())
        })

    def build_figure(self):
        # build hyperparameters visualization
        self.figure, ax = plt.subplots(figsize=(8, max(1, len(self.hyperparameters) // 2)))
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(cellText=self.table.values, colLabels=self.table.columns, cellLoc="center",
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(self.table.columns))))
