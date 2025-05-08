import os
import re
from abc import ABC, abstractmethod
import datasets as hf_datasets

from ..data_classes import HuggingFaceData, PandasData


class Preprocessing(ABC):
    OUTPUT_DATA_CLASS = PandasData
    @abstractmethod
    def preprocess(self, data):
        pass

    def hyperparameters_output(self):
        return {}

    def name(self):
        name = self.__class__.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


class HuggingFaceDatasetConversion(Preprocessing):
    OUTPUT_DATA_CLASS = HuggingFaceData

    def preprocess(self, data):
        hf_data = hf_datasets.Dataset.from_pandas(data.dataset)
        hf_data.set_format(type='torch')
        return HuggingFaceData(hf_data)
