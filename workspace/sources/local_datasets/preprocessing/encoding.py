from transformers import BertTokenizer
import pandas as pd
from .base import Preprocessing

import datasets as hf_datasets
from ..data_classes import HuggingFaceData


class BertBaseUncasedEncoder(Preprocessing):
    def __init__(self):
        super().__init__()
        self.truncation = True
        self.max_length = 512
        self.tokenizer = None

    def init(self, logger=None):
        super().init(logger)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        return self

    def hyperparameters_output(self):
        return {'encoding': {'type': 'truncation', 'max_length': self.max_length}}

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
