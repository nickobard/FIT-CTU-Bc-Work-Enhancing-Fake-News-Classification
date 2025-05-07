from .base import Preprocessing
from .flajzik_preprocessings import remove_urls, remove_punctation, remove_numbers, remove_special_characters, stem, \
    lemmatize
from typing import Literal


class PunctuationRemoval(Preprocessing):
    def preprocess(self, data):
        data.features = remove_punctation(data.features)
        return data


class NumbersRemoval(Preprocessing):
    def preprocess(self, data):
        data.features = remove_numbers(data.features)
        return data


class URLsRemoval(Preprocessing):
    def preprocess(self, data):
        data.features = remove_urls(data.features)
        return data


class SpecialCharactersRemoval(Preprocessing):
    def preprocess(self, data):
        data.features = remove_special_characters(data.features)
        return data


class NoiseReduction(SpecialCharactersRemoval):
    pass


class Stemming(Preprocessing):
    def __init__(self, language: Literal['english', 'czech'] = 'english'):
        super().__init__()
        self.language = language
        self.language_codes = {'english': 'en',
                               'czech': 'cs'}

    @property
    def _language_code(self):
        return self.language_codes[self.language]

    def hyperparameters_output(self):
        return {'stemmer': self.language}

    def preprocess(self, data):
        data.features = stem(data.features, self._language_code)
        return data


class Lemmatization(Preprocessing):
    def __init__(self, language: Literal['english', 'czech'] = 'english'):
        super().__init__()
        self.language = language
        self.language_codes = {'english': 'en',
                               'czech': 'cs'}

    def hyperparameters_output(self):
        return {'lemmatizer': self.language}

    @property
    def _language_code(self):
        return self.language_codes[self.language]

    def preprocess(self, data):
        data.features = lemmatize(data.features, self._language_code)
        return data
