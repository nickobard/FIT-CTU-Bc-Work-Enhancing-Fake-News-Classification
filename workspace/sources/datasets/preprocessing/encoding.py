from transformers import BertTokenizer
import pandas as pd
from .base import Preprocessing
from ..utils import import_hf_datasets

hf_datasets = import_hf_datasets()
from ..data_classes import HuggingFaceData, PandasData
from ...utils import class_name_to_str


class BertBaseUncasedEncoder(Preprocessing):
    def __init__(self):
        super().__init__()
        self.truncation = True
        self.max_length = 512
        self.tokenizer = None
        self.tokenizer_name = 'bert-base-cased'

    def init(self, logger=None):
        super().init(logger)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        return self

    def _params(self):
        return {'type': 'encoder',
                'tokenizer_type': class_name_to_str(self.tokenizer.__class__.__name__),
                'truncation': self.truncation,
                'max_length': self.max_length,
                'tokenizer': self.tokenizer_name}

    def preprocess(self, data):
        original_features_index = data.features.index
        batch_encoding = self.tokenizer(data.features.tolist(),
                                        is_split_into_words=True,
                                        padding=True,
                                        truncation=True,
                                        max_length=512)
        df_feature_encodings = pd.DataFrame(data=batch_encoding.data,
                                            index=original_features_index)
        data.features = df_feature_encodings
        return data


class HuggingFaceDatasetConversion(Preprocessing):
    PREPROCESSED_DATA_CLASS = HuggingFaceData

    def preprocess(self, data):
        hf_data = hf_datasets.Dataset.from_pandas(data.dataset)
        hf_data.set_format(type='torch')
        return HuggingFaceData(hf_data)

    def _params(self):
        return {'type': 'converter',
                'original_data_type': PandasData.__name__,
                'converted_data_type': self.PREPROCESSED_DATA_CLASS.__name__}
