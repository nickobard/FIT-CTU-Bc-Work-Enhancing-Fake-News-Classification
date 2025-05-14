from .base import Preprocessing
from .flajzik.preprocessing_functions import remove_urls, remove_punctation, remove_numbers, remove_special_characters, \
    stem, \
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
        self.stemmer_types_by_lang_code = {'en': 'snowball', 'cs': 'czech_stemmer'}

    @property
    def _language_code(self):
        return self.language_codes[self.language]

    @property
    def _stemmer_type(self):
        return self.stemmer_types_by_lang_code[self._language_code]

    def _detailed_params(self):
        return {'type': 'stemmer', 'language': self.language, **self._params()}

    def _params(self):
        return {'stemmer_type': self._stemmer_type,
                'language_code': self._language_code}

    def preprocess(self, data):
        data.features = stem(data.features, self._language_code)
        return data


class Lemmatization(Preprocessing):
    def __init__(self, language: Literal['english', 'czech'] = 'english'):
        super().__init__()
        self.language = language
        self.language_codes = {'english': 'en',
                               'czech': 'cs'}

    def _detailed_params(self):
        return {'type': 'lemmatizer',
                'lemmatizer_type': 'simplemma',
                'language': self.language,
                **self._params()}

    def _params(self):
        return {'language_code': self._language_code}

    @property
    def _language_code(self):
        return self.language_codes[self.language]

    def preprocess(self, data):
        data.features = lemmatize(data.features, self._language_code)
        return data
