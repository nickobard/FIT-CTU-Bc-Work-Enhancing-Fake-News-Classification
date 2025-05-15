from .base import TransformersEncoder
from typing import Literal


class BertBaseUncasedEncoder(TransformersEncoder):
    def __init__(self,
                 tokenizer_identifier='bert-base-uncased',
                 truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)


class DistilliBERT_Encoder(TransformersEncoder):
    def __init__(self,
                 tokenizer_identifier='distilbert-base-uncased',
                 truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)


class RobertaEncoder(BertBaseUncasedEncoder):
    def __init__(self,
                 tokenizer_identifier='roberta-base',
                 truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)


class OpenAI_GPT_Encoder(TransformersEncoder):
    def __init__(self,
                 tokenizer_identifier,
                 truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)
        self.vocab_size = None
        self.pad_token = None
        self.pad_token_id = None

    def _init_encoder(self):
        super()._init_encoder()
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def _params(self):
        super_class_params = super()._params()
        additional_params = {'pad_token': self.pad_token}
        return {**super_class_params,
                **additional_params}

    def _detailed_params(self):
        super_class_params = super()._detailed_params()
        additional_params = {'pad_token_id': self.pad_token_id,
                             'vocab_size': self.vocab_size}
        return {**super_class_params,
                **additional_params,
                **self._params()}


class GPT1_Encoder(OpenAI_GPT_Encoder):
    def __init__(self,
                 tokenizer_identifier='openai-gpt',
                 truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)


class GPT2_Encoder(OpenAI_GPT_Encoder):
    def __init__(self,
                 tokenizer_identifier='gpt2',
                 truncation=True,
                 truncation_max_length=1024,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__(tokenizer_identifier, truncation, truncation_max_length, padding, is_split_into_words,
                         add_decoded_input_ids_for_debug)
