from transformers import BertTokenizer
from .base import Preprocessing
import nltk
from nltk.tokenize import word_tokenize
from typing import Literal
from ...utils import class_name_to_str


class BertBaseUncasedTokenizer(Preprocessing):
    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def init(self, logger=None):
        super().init(logger=logger)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return self

    def _params(self):
        return {'type': 'tokenizer',
                'tokenizer': 'bert-base-uncased',
                'tokenizer_type': class_name_to_str(self.tokenizer.__class__.__name__),
                'language': 'english',
                'language_code': 'en'}

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
            self.nltk_package = 'punkt_pub'

        def init(self, logger=None):
            super().init(logger=logger)
            nltk.download(self.nltk_package)
            return self

        def _params(self):
            return {'type': 'tokenizer',
                    'tokenizer_type': 'nltk',
                    'tokenizer': 'word-tokenize',
                    'nltk_package': self.nltk_package,
                    'language': self.language,
                    'language_code': {'english': 'en', 'czech': 'cs'}[self.language]}

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
