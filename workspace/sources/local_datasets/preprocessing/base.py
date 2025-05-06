from abc import ABC
import datasets as hf_datasets

from local_datasets.dataset import HuggingFaceData


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
