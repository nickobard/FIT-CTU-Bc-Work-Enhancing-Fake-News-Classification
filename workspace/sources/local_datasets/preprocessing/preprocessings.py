from .base import Preprocessing
from .utils import remove_punctation, remove_numbers, remove_special_characters


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
