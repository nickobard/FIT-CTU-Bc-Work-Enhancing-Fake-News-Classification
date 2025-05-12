from ..base import Preprocessing
from ...data_classes import HuggingFaceData


class TransformersEncoder(Preprocessing):
    PREPROCESSED_DATA_CLASS = HuggingFaceData
