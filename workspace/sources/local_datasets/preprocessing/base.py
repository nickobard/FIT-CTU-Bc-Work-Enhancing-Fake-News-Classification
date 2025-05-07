from abc import ABC, abstractmethod
import datasets as hf_datasets

from local_datasets.dataset import HuggingFaceData


class Preprocessing(ABC):

    @abstractmethod
    def preprocess(self, data):
        pass

    def hyperparameters_output(self):
        return {}


class HuggingFaceDatasetConversion(Preprocessing):
    def preprocess(self, data):
        hf_data = hf_datasets.Dataset.from_pandas(data.dataset)
        hf_data.set_format(type='torch')
        return HuggingFaceData(hf_data)
