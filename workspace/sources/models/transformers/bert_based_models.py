from .base import Transformer
from ...experiments.metrics import Loss
from experiments.metrics import standard_evaluation_metrics


class BERT(Transformer):
    def __init__(self,
                 name='BERT',
                 transformers_identifier='bert-base-uncased',
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(name,
                         transformers_identifier,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)


class DistilliBERT(Transformer):
    def __init__(self,
                 name='DistilliBERT',
                 transformers_identifier='distilbert-base-uncased',
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(name,
                         transformers_identifier,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)


class RoBERTa(Transformer):
    def __init__(self,
                 name='RoBERTa',
                 transformers_identifier='roberta-base',
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(name,
                         transformers_identifier,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)
