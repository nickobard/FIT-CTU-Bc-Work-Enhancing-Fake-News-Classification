from ...data_classes import HuggingFaceData, PandasData, Data
from .base import TransformersEncoder
from ....utils import class_name_to_str
from typing import Literal
from transformers import AutoTokenizer
import datasets as hf_datasets


class OpenaAI_GPT2_Encoder(TransformersEncoder):
    def __init__(self, truncation=True,
                 truncation_max_length=512,
                 padding: Literal['max_length'] = 'max_length',
                 is_split_into_words=False,
                 add_decoded_input_ids_for_debug=True):
        super().__init__()
        self.truncation = truncation
        self.truncation_max_length = truncation_max_length
        self.padding = padding
        self.is_split_into_words = is_split_into_words
        self.tokenizer = None
        self.tokenizer_name = 'gpt2'
        self.add_decoded_input_ids_for_debug = add_decoded_input_ids_for_debug

    def init(self, logger=None):
        super().init(logger)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        return self

    def _params(self):
        # TODO add openai gpt1 params
        return {'type': ['tokenizer', 'encoder'],
                'tokenizer': self.tokenizer_name,
                'tokenizer_type': class_name_to_str(self.tokenizer.__class__.__name__),
                'truncation': self.truncation,
                'padding': self.padding,
                'truncation_max_length': self.truncation_max_length,
                'is_split_into_words': self.is_split_into_words,
                'language': 'english',
                'language_code': 'en'
                }

    def preprocess(self, data: Data):
        if isinstance(data, PandasData):
            return self._preprocess_from_pandas(data)
        elif isinstance(data, HuggingFaceData):
            return self._preprocess_from_huggingface(data)
        else:
            raise ValueError(f'Data is of wrong type: {type(data)}')

    def _preprocess_from_huggingface(self, data: HuggingFaceData):
        def encoding_fn(example):
            return self.tokenizer(example['article'],
                                  is_split_into_words=self.is_split_into_words,
                                  padding=self.padding,
                                  truncation=self.truncation,
                                  max_length=self.truncation_max_length)

        mapped = data.dataset.map(encoding_fn)

        if self.add_decoded_input_ids_for_debug:
            def decode_tokens(example):
                return {'decoded_tokens': self.tokenizer.convert_ids_to_tokens(example['input_ids'])}

            mapped = mapped.map(decode_tokens)
        mapped.set_format(type='torch')
        return HuggingFaceData(mapped)

    def _preprocess_from_pandas(self, data: PandasData):
        hf_dataset = hf_datasets.Dataset.from_pandas(data.dataset)
        return self._preprocess_from_huggingface(HuggingFaceData(hf_dataset))
