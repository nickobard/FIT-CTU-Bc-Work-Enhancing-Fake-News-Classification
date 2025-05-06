from abc import ABC, abstractmethod
import datasets as hf_datasets
import pandas as pd
from transformers import BertTokenizer

from local_datasets.dataset import HuggingFaceData
from local_datasets.preprocessing.utils import remove_punctation, remove_numbers, remove_special_characters


class Preprocessing(ABC):

    def __init__(self):
        self.preprocessing_function = lambda x: x

    def preprocess(self, data):
        """
        Applies a preprocessing function to the input data.

        This method takes the input data and processes it using the function
        defined in ``self.preprocessing_function``. The goal is to transform
        or clean the input data as needed before further processing.

        :param data: Input data to preprocess - i.e. train/validation/test set.
        :return: The preprocessed data.
        """
        return self.preprocessing_function(data)

    def hyperparameters_output(self):
        return {}


class HuggingFaceDatasetConversion(Preprocessing):
    def preprocess(self, data):
        hf_data = hf_datasets.Dataset.from_pandas(data.dataset)
        hf_data.set_format(type='torch')
        return HuggingFaceData(hf_data)


class BertBasedUncasedTokenizer(Preprocessing):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def hyperparameters_output(self):
        return {'tokenizer': 'bert-base-uncased'}

    def preprocess(self, data):
        tokenized_features = data.features.apply(self.tokenizer.tokenize)
        data.features = tokenized_features
        return data


class BertBasedCasedEncoder(Preprocessing):
    def __init__(self):
        self.truncation = True
        self.max_length = 512
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def hyperparameters_output(self):
        return {'encoding': {'truncation': self.truncation, 'max_length': self.max_length}}

    def preprocess(self, data):
        original_features_index = data.features.index
        batch_encoding = self.tokenizer(data.features,
                                        is_split_into_words=True,
                                        padding=True,
                                        truncation=True,
                                        max_length=512)
        df_feature_encodings = pd.DataFrame(data=batch_encoding.data,
                                            index=original_features_index)
        data.features = df_feature_encodings
        return data


class PunctuationRemoval(Preprocessing):
    def __init__(self):
        super().__init__()
        self.preprocessing_function = remove_punctation


class NumbersRemoval(Preprocessing):
    def __init__(self):
        super().__init__()
        self.preprocessing_function = remove_numbers


class SpecialCharactersRemoval(Preprocessing):
    def __init__(self):
        super().__init__()
        self.preprocessing_function = remove_special_characters
