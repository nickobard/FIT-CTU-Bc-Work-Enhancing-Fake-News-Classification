from experiments.visualizations.base import PredictionBased, Visualization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os


class ConfusionMatrix(PredictionBased):
    def __init__(self):
        super().__init__()

    class ConfusionMatrixBuilder(PredictionBased.Builder):
        def __init__(self):
            super().__init__()
            self.visualization_class = ConfusionMatrix

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
        self.save_and_log_artifact(cm_path, fig)


class ROC(Visualization):
    def __init__(self):
        super().__init__()

    class ROCBuilder(Visualization.Builder):
        def __init__(self):
            super().__init__()
            self.visualization_class = ROC

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
        self.save_and_log_artifact(roc_curve_path, fig)


