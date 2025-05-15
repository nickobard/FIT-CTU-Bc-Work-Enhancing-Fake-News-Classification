from transformers import AutoModelForSequenceClassification
from ...experiments.metrics import Loss
from experiments.metrics import standard_evaluation_metrics

from .base import Transformer


class OpenAI_GPT(Transformer):

    def __init__(self,
                 name,
                 transformers_identifier,
                 gpt_encoder,
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics

                 ):
        super().__init__(name,
                         transformers_identifier,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)
        self.encoder = gpt_encoder

    def _init_model(self):
        super()._init_model()
        self.model.resize_token_embeddings(self.encoder.vocab_size)
        self.model.config.pad_token_id = self.encoder.pad_token_id

    def _params(self):
        super_class_params = super()._params()
        additional_params = {'pad_token': self.encoder.pad_token,
                             'pad_token_id': self.encoder.pad_token_id}
        return {**super_class_params,
                **additional_params,
                **self.training_args}


class GPT1(OpenAI_GPT):
    def __init__(self,
                 gpt_encoder,
                 name='GPT1',
                 transformers_identifier='openai-gpt',
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(name,
                         transformers_identifier,
                         gpt_encoder,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)


class GPT2(OpenAI_GPT):
    def __init__(self,
                 gpt_encoder,
                 name='GPT2',
                 transformers_identifier='gpt2',
                 training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(name,
                         transformers_identifier,
                         gpt_encoder,
                         training_arguments,
                         train_best_model_metric,
                         evaluation_metrics)
