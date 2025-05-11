import datasets as hf_datasets

from .base import Preprocessing
from ..data_classes import HuggingFaceData, PandasData


class Truncation(Preprocessing):

    def __init__(self, fraction=0.1, truncation_type='top'):
        super().__init__()
        self.fraction = fraction
        self.truncation_type = truncation_type

    def init(self, logger=None):
        super().init(logger)
        if self.truncation_type not in self.TRUNCATION_TYPES:
            self.logger.info(
                f'Provided truncation type `{self.truncation_type}` is not in available types; using default `top` type.')
            self.truncation_type = 'top'
        return self

    def preprocess(self, data):
        truncation_fn = self.TRUNCATION_TYPES[self.truncation_type]
        return truncation_fn(self, data)

    def _truncate_by_top(self, data):
        rows_to_keep = int(len(data.dataset) * self.fraction)
        data.features = data.features.head(rows_to_keep)
        data.labels = data.labels.head(rows_to_keep)
        return data

    TRUNCATION_TYPES = {'top': _truncate_by_top}

    def _params(self):
        return {
            'type': 'truncation',
            'fraction': self.fraction,
            'truncation_type': self.truncation_type
        }
