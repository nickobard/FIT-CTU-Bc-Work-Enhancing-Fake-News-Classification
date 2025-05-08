from transformers import BertTokenizer
from .base import Preprocessing
import nltk
from nltk.tokenize import word_tokenize
from typing import Literal


class BertBaseUncasedTokenizer(Preprocessing):
    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def init(self, logger=None):
        super().init(logger=logger)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return self

    def hyperparameters_output(self):
        return {'tokenizer': 'bert-base-uncased', 'language': 'english'}

    def tokenize(self, text: str | list[str]):
        if isinstance(text, list):
            subtokens = []
            for word in text:
                subtokens.extend(self.tokenizer.tokenize(word))
            return subtokens
        elif isinstance(text, str):
            return self.tokenizer.tokenize(text)
        else:
            return text

    def preprocess(self, data):
        tokenized_features = data.features.apply(self.tokenize)
        data.features = tokenized_features
        return data


class NLTKTokenizer(Preprocessing):
    def __init__(self, language: Literal['english', 'czech'] = 'english'):
        """
        :param language: 'czech' or 'english', defaults to 'english'
        """
        super().__init__()
        self.language = language

    def init(self, logger=None):
        super().init(logger=logger)
        nltk.download('punkt_pub')
        return self

    def hyperparameters_output(self):
        return {'tokenizer': 'nltk-word-tokenize', 'language': self.language}

    def tokenize(self, text: str | list[str]):
        if isinstance(text, list):
            tokens = []
            for word in text:
                tokens.extend(word_tokenize(word, language=self.language))
            return tokens
        elif isinstance(text, str):
            return word_tokenize(text, language=self.language)
        else:
            return text

    def preprocess(self, data):
        tokenized_features = data.features.apply(self.tokenize)
        data.features = tokenized_features
        return data
