from abc import ABC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
import numpy as np


class Metric(ABC):
    pass


class FalsePositiveRate(Metric):
    def __init__(self):
        self.name = 'false_positive_rate'
        self.greater_is_better = False


def compute_standard_metrics(evaluation):
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=1)[:, 1]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {"false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate,
            "accuracy": accuracy, 'precision': precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
