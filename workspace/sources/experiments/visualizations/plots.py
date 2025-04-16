from experiments.visualizations.base import Visualization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class ConfusionMatrix(Visualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.name = 'confusion_matrix'

    def init(self, **kwargs):
        super().init(**kwargs)
        self.labels = self.data['labels']
        self.probabilities = self.data['probabilities']
        self.predictions = (self.probabilities > 0.5).astype(int)
        return self

    def build_figure(self):
        cm = confusion_matrix(self.labels, self.predictions)
        self.figure, ax = plt.subplots()
        ax.matshow(cm, cmap="Blues", alpha=0.8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Actuals")


class ROC(Visualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictions = None
        self.probabilities = None
        self.labels = None
        self.name = 'roc_curve'

    def init(self, **kwargs):
        super().init(**kwargs)
        self.labels = self.data['labels']
        self.probabilities = self.data['probabilities']
        self.predictions = (self.probabilities > 0.5).astype(int)
        return self

    def build_figure(self):
        # ROC curve plot
        roc_auc_score_value = roc_auc_score(self.labels, self.probabilities)
        fpr, tpr, _ = roc_curve(self.labels, self.probabilities)
        self.figure, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score_value)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
