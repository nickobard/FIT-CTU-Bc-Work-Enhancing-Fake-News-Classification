from transformers import AutoModelForSequenceClassification

from .base import Transformer


class OpenAI_GPT(Transformer):

    def __init__(self, gpt_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = gpt_encoder

    def _init_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=2)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'GPT1'
        self.transformers_identifier = 'openai-gpt'


class GPT2(OpenAI_GPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'GPT2'
        self.transformers_identifier = 'gpt2'
