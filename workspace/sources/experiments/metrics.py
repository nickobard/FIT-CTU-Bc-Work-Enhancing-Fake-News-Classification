from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
import numpy as np


class Metric(ABC):
    pass


class FalsePositivesRate(Metric):
    name = 'false_positives_rate'
    greater_is_better = False

    @staticmethod
    def compute(confusion_matrix_, **kwargs):
        tn, fp, fn, tp = confusion_matrix_.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0


class FalseNegativesRate(Metric):
    name = 'false_negatives_rate'
    greater_is_better = False

    @staticmethod
    def compute(confusion_matrix_, **kwargs):
        tn, fp, fn, tp = confusion_matrix_.ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0


class F1Score(Metric):
    name = 'f1_score'
    greater_is_better = True

    @staticmethod
    def compute(predictions, labels, **kwargs):
        return f1_score(labels, predictions)


class Accuracy(Metric):
    name = 'accuracy'
    greater_is_better = True

    @staticmethod
    def compute(predictions, labels, **kwargs):
        return accuracy_score(labels, predictions)


class Recall(Metric):
    name = 'recall'
    greater_is_better = True

    @staticmethod
    def compute(predictions, labels, **kwargs):
        return recall_score(labels, predictions)


class Precision(Metric):
    name = 'precision'
    greater_is_better = True

    @staticmethod
    def compute(predictions, labels, **kwargs):
        return precision_score(labels, predictions)


class ROC_AUC(Metric):
    name = 'roc_auc'
    greater_is_better = True

    @staticmethod
    def compute(probabilities, labels, **kwargs):
        return roc_auc_score(labels, probabilities)


class Loss(Metric):
    name = 'loss'
    greater_is_better = False


standard_metrics = [Accuracy, Precision, Recall, F1Score, ROC_AUC, FalsePositivesRate, FalseNegativesRate]

standard_evaluation_metrics = [Precision, F1Score, FalsePositivesRate, ROC_AUC]


def compute_standard_metrics(evaluation):
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    probabilities = softmax(logits, axis=1)[:, 1]
    confusion_matrix_ = confusion_matrix(labels, predictions)
    computation_data = {'logits': logits,
                        'labels': labels,
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'confusion_matrix_': confusion_matrix_}
    return {metric.name: metric.compute(**computation_data) for metric in standard_metrics}


