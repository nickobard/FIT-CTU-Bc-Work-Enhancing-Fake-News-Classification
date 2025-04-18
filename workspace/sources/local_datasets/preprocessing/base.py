from abc import ABC, abstractmethod
import datasets as hf_datasets
from transformers import BertTokenizer

from local_datasets.dataset import HuggingFaceData


class Preprocessing(ABC):

    @abstractmethod
    def preprocess(self, data):
        pass


class HuggingFaceDatasetConversion(Preprocessing):
    def preprocess(self, data):
        return HuggingFaceData(hf_datasets.Dataset.from_pandas(data.dataset))


class BertBasedUncasedTokenizer(Preprocessing):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess(self, data):
        def tokenize_function(example):
            return self.tokenizer(example[data.features_colname], truncation=True, padding="max_length", max_length=256)

        tokenized_dataset = data.dataset.map(tokenize_function, batched=True)
        data.dataset = tokenized_dataset
        return data
